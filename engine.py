# engine.py
# Core TTS model loading and speech generation logic.
# Supports Dia 1.6B (original) and Dia 2 models (1B, 2B) with hot-swappable model switching.

import gc
import logging
import time
import os
import threading
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- Defensive Imports: Dia 1 (original) ---
try:
    from dia.model import (
        Dia as Dia1Model,
        ComputeDtype,
        DEFAULT_SAMPLE_RATE as DIA1_SAMPLE_RATE,
    )
    from dia.config import DiaConfig

    DIA1_AVAILABLE = True
except ImportError:
    Dia1Model = None
    ComputeDtype = None
    DiaConfig = None
    DIA1_SAMPLE_RATE = 44100
    DIA1_AVAILABLE = False
    logging.info("Dia 1 (dia) package not available. Dia 1.6B model will be unavailable.")

# --- Defensive Imports: Dia 2 ---
try:
    from dia2 import Dia2, GenerationConfig, SamplingConfig, PrefixConfig, GenerationResult

    DIA2_AVAILABLE = True
except ImportError:
    Dia2 = None
    GenerationConfig = None
    SamplingConfig = None
    PrefixConfig = None
    GenerationResult = None
    DIA2_AVAILABLE = False
    logging.info("Dia 2 (dia2) package not available. Dia 2 models will be unavailable.")

# Import configuration getters from our project's config.py
from config import (
    get_model_repo_id,
    get_model_cache_path,
    get_reference_audio_path,
    get_model_config_filename,
    get_model_weights_filename,
    get_gen_default_seed,
    get_whisper_model_name,
)

# Import text splitting utility and other helpers
from utils import (
    chunk_text_by_sentences,
    PerformanceMonitor,
    trim_lead_trail_silence,
    fix_internal_silence,
    remove_long_unvoiced_segments,
    _generate_transcript_with_whisper,
)

logger = logging.getLogger(__name__)

# --- Model Registry ---
# Maps friendly selector names to model metadata.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dia-1.6b": {
        "repo_id": "ttj/dia-1.6b-safetensors",
        "display_name": "Dia 1.6B (Original)",
        "params": "1.6B",
        "model_type": "dia1",
        "sample_rate": 44100,
        "voices": ["dialogue", "single_s1", "single_s2", "clone", "predefined"],
        "default_voice": "predefined",
        "supports_cloning": True,
        "cloning_method": "audio_prompt",
    },
    "dia2-1b": {
        "repo_id": "nari-labs/Dia2-1B",
        "display_name": "Dia 2 — 1B (Streaming)",
        "params": "1B",
        "model_type": "dia2",
        "sample_rate": None,  # Read from model at runtime
        "voices": ["dialogue", "single_s1", "single_s2", "clone", "predefined"],
        "default_voice": "predefined",
        "supports_cloning": True,
        "cloning_method": "prefix_speaker",
    },
    "dia2-2b": {
        "repo_id": "nari-labs/Dia2-2B",
        "display_name": "Dia 2 — 2B (High Quality)",
        "params": "2B",
        "model_type": "dia2",
        "sample_rate": None,  # Read from model at runtime
        "voices": ["dialogue", "single_s1", "single_s2", "clone", "predefined"],
        "default_voice": "predefined",
        "supports_cloning": True,
        "cloning_method": "prefix_speaker",
    },
}

# Build reverse lookup: repo_id -> selector key
_REPO_TO_SELECTOR: Dict[str, str] = {}
for _sel, _info in MODEL_REGISTRY.items():
    _REPO_TO_SELECTOR[_info["repo_id"]] = _sel


def resolve_selector(config_value: str) -> str:
    """
    Resolves a config model.repo_id value to a registry selector key.
    Accepts either a selector key directly or a full HuggingFace repo_id.
    Returns the selector key, or falls back to 'dia-1.6b' if unknown.
    """
    if config_value in MODEL_REGISTRY:
        return config_value
    if config_value in _REPO_TO_SELECTOR:
        return _REPO_TO_SELECTOR[config_value]
    lower = config_value.lower().strip()
    for sel in MODEL_REGISTRY:
        if sel.lower() == lower:
            return sel
    for repo_id, sel in _REPO_TO_SELECTOR.items():
        if repo_id.lower() == lower:
            return sel
    logger.warning(
        f"Unknown model selector '{config_value}'. "
        f"Valid selectors: {list(MODEL_REGISTRY.keys())}. "
        f"Defaulting to 'dia-1.6b'."
    )
    return "dia-1.6b"


# --- Global Module Variables ---
dia_model: Optional[Any] = None  # Dia1Model or Dia2 instance
model_config_instance: Optional[Any] = None
model_device: Optional[torch.device] = None
MODEL_LOADED: bool = False
EXPECTED_SAMPLE_RATE: int = 44100

# Track which model is loaded
loaded_model_selector: Optional[str] = None
loaded_model_type: Optional[str] = None  # "dia1" or "dia2"

# --- Async Loading & Cancellation ---
_load_lock = threading.Lock()
_cancel_event = threading.Event()
_load_thread: Optional[threading.Thread] = None

# Download progress tracking for UI status modal
_download_status: Dict[str, Any] = {
    "active": False,
    "phase": "",
    "detail": "",
    "progress_pct": 0,
    "error": None,
}


def _check_cancelled():
    """Raises RuntimeError if model loading has been cancelled."""
    if _cancel_event.is_set():
        raise RuntimeError("Model loading cancelled by user.")


def _update_download_status(phase: str, detail: str = "", progress_pct: int = 0, error: str = None):
    """Update the download status for the UI to poll."""
    global _download_status
    _download_status = {
        "active": error is None and phase != "complete",
        "phase": phase,
        "detail": detail,
        "progress_pct": progress_pct,
        "error": error,
    }


def get_download_status() -> Dict[str, Any]:
    """Returns the current download/loading status for UI polling."""
    return dict(_download_status)


def is_loading() -> bool:
    """Returns True if a model is currently being loaded in the background."""
    return _load_thread is not None and _load_thread.is_alive()


def get_model_info() -> Dict[str, Any]:
    """Returns information about the currently loaded model."""
    if loaded_model_selector and loaded_model_selector in MODEL_REGISTRY:
        reg = MODEL_REGISTRY[loaded_model_selector]
        return {
            "loaded": MODEL_LOADED,
            "selector": loaded_model_selector,
            "repo_id": reg["repo_id"],
            "display_name": reg["display_name"],
            "params": reg["params"],
            "model_type": reg["model_type"],
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "voices": reg["voices"],
            "default_voice": reg["default_voice"],
            "supports_cloning": reg["supports_cloning"],
            "cloning_method": reg["cloning_method"],
            "device": str(model_device) if model_device else None,
        }
    return {
        "loaded": MODEL_LOADED,
        "selector": loaded_model_selector,
        "repo_id": None,
        "display_name": None,
        "params": None,
        "model_type": loaded_model_type,
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "voices": [],
        "default_voice": None,
        "supports_cloning": False,
        "cloning_method": None,
        "device": str(model_device) if model_device else None,
    }


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """Returns the full model registry for the UI dropdown, filtered by availability."""
    result = {}
    for k, v in MODEL_REGISTRY.items():
        available = True
        if v["model_type"] == "dia1" and not DIA1_AVAILABLE:
            available = False
        if v["model_type"] == "dia2" and not DIA2_AVAILABLE:
            available = False
        result[k] = {
            "display_name": v["display_name"],
            "params": v["params"],
            "model_type": v["model_type"],
            "voices": v["voices"],
            "default_voice": v["default_voice"],
            "supports_cloning": v["supports_cloning"],
            "cloning_method": v["cloning_method"],
            "available": available,
        }
    return result

# --- Model Loading ---


def get_device() -> torch.device:
    """Determines the optimal torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        logger.info("CUDA is available, using GPU.")
        torch.cuda.empty_cache()
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS support check might need refinement based on PyTorch version
        try:
            # Simple check if MPS device can be created and used
            test_tensor = torch.tensor([1.0]).to("mps")
            if test_tensor.device.type == "mps":
                logger.info("MPS is available, using Apple Silicon GPU.")
                return torch.device("mps")
            else:
                raise RuntimeError("MPS device creation failed.")
        except Exception as e:
            logger.info(f"MPS not fully available ({e}), falling back to CPU.")
            return torch.device("cpu")
    else:
        logger.info("CUDA and MPS not available, using CPU.")
        return torch.device("cpu")


def get_compute_dtype(device: torch.device, weights_filename: str) -> str:
    """
    Determines the recommended compute dtype string based on device capabilities
    AND the type of model weights indicated by the filename.

    Args:
        device: The torch.device the model will run on (e.g., torch.device('cuda')).
        weights_filename: The filename of the model weights being loaded (e.g., "dia-v0_1_bf16.safetensors").

    Returns:
        A string representing the compute dtype (e.g., "bfloat16", "float16", "float32").
    """
    # Determine if the filename suggests a BF16 model by checking for 'bf16' substring
    is_bf16_model = "bf16" in weights_filename.lower()
    logger.info(
        f"Analyzing weights filename '{weights_filename}': Detected model type appears to be {'BF16' if is_bf16_model else 'Non-BF16 (likely FP32/FP16)'}"
    )

    # Check if ComputeDtype enum was successfully imported
    if ComputeDtype is None:
        logger.warning(
            "ComputeDtype enum not available from dia.model (import failed?). Defaulting compute dtype selection to basic fallback."
        )
        # --- Fallback logic without the enum ---
        # This provides basic functionality if the enum import fails.
        if device.type == "cuda":
            # Check hardware capabilities directly
            supports_bf16_fallback = torch.cuda.is_bf16_supported()
            compute_capability_fallback = torch.cuda.get_device_capability(device)
            supports_fp16_fallback = (
                compute_capability_fallback[0] >= 7
            )  # Tensor Cores needed for efficient FP16

            if is_bf16_model:
                if supports_bf16_fallback:
                    logger.info(
                        "Fallback: BF16 model and BF16 supported. Using 'bfloat16'."
                    )
                    return "bfloat16"
                elif supports_fp16_fallback:
                    logger.warning(
                        "Fallback: BF16 model, BF16 not supported. Using 'float16'."
                    )
                    return "float16"
                else:
                    logger.warning(
                        "Fallback: BF16 model, BF16/FP16 not supported. Using 'float32'."
                    )
                    return "float32"
            else:  # Non-BF16 model
                logger.info(
                    "Fallback: Non-BF16 model. Using 'float32' for compatibility."
                )
                return "float32"  # Prioritize FP32 for non-BF16 models

        elif device.type == "mps":
            logger.info("Fallback: MPS device. Using 'float16'.")
            return "float16"  # FP16 is generally preferred on MPS
        else:  # CPU
            logger.info("Fallback: CPU device. Using 'float32'.")
            return "float32"  # FP32 is standard for CPU

    # --- Preferred logic using the ComputeDtype enum ---
    if device.type == "cuda":
        supports_bf16 = torch.cuda.is_bf16_supported()
        # Check device capability for FP16 (requires compute capability >= 7.0 for efficient use)
        compute_capability = torch.cuda.get_device_capability(device)
        supports_fp16 = compute_capability[0] >= 7

        if is_bf16_model:
            if supports_bf16:
                logger.info(
                    "BF16 model detected and CUDA device supports bfloat16. Using BF16 compute dtype."
                )
                return ComputeDtype.BFLOAT16.value  # Return string value "bfloat16"
            elif supports_fp16:
                logger.warning(
                    "BF16 model detected, but CUDA device does NOT support bfloat16. Falling back to FP16 compute dtype."
                )
                return ComputeDtype.FLOAT16.value  # Return string value "float16"
            else:
                logger.warning(
                    "BF16 model detected, but CUDA device supports neither bfloat16 nor efficient FP16. Falling back to FP32 compute dtype."
                )
                return ComputeDtype.FLOAT32.value  # Return string value "float32"
        else:  # Not a BF16 model (e.g., dia-v0_1.safetensors or dia-v0_1.pth)
            # ** FIX: If the model isn't explicitly BF16, prioritize FP32 for accuracy, **
            # ** regardless of hardware support for lower precisions. **
            logger.info(
                "Non-BF16 model weights detected. Using FP32 compute dtype on CUDA for maximum compatibility and accuracy."
            )
            return ComputeDtype.FLOAT32.value  # Return string value "float32"

    elif device.type == "mps":
        # MPS generally works best with FP16 or FP32. Defaulting to FP16 for potential performance benefits.
        # If issues arise with non-BF16 models on MPS, consider changing this to FP32.
        logger.info(
            "MPS device detected. Using FP16 compute dtype as a general recommendation."
        )
        return ComputeDtype.FLOAT16.value  # Return string value "float16"

    else:  # CPU
        logger.info("CPU device detected. Using FP32 compute dtype.")
        return ComputeDtype.FLOAT32.value  # Return string value "float32"


def load_model():
    """
    Loads a TTS model based on the current config selector.
    Supports both Dia 1 (original) and Dia 2 (1B/2B) models.
    Downloads model files from HuggingFace if not cached.
    """
    global dia_model, model_config_instance, model_device, MODEL_LOADED, EXPECTED_SAMPLE_RATE
    global loaded_model_selector, loaded_model_type

    if MODEL_LOADED:
        logger.info("Model already loaded.")
        _update_download_status("complete", "Model already loaded.", 100)
        return True

    try:
        # Resolve the model selector from config
        config_value = get_model_repo_id()
        selector = resolve_selector(config_value)
        reg = MODEL_REGISTRY[selector]
        model_type = reg["model_type"]
        repo_id = reg["repo_id"]

        # Check availability
        if model_type == "dia1" and not DIA1_AVAILABLE:
            raise ImportError("Dia 1 (dia) package is not installed. Cannot load Dia 1.6B model.")
        if model_type == "dia2" and not DIA2_AVAILABLE:
            raise ImportError("Dia 2 (dia2) package is not installed. Cannot load Dia 2 models.")

        cache_path = get_model_cache_path()
        model_device = get_device()

        logger.info(f"Loading model: {reg['display_name']} ({repo_id})")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Cache directory: {cache_path}")
        logger.info(f"  Target device: {model_device}")

        os.makedirs(cache_path, exist_ok=True)
        start_time = time.time()

        if model_type == "dia1":
            _load_dia1_model(repo_id, cache_path)
        elif model_type == "dia2":
            _load_dia2_model(repo_id, cache_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        loaded_model_selector = selector
        loaded_model_type = model_type
        MODEL_LOADED = True

        end_time = time.time()
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")
        _update_download_status("complete", f"{reg['display_name']} loaded successfully.", 100)
        return True

    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Model loading cancelled: {e}")
            _update_download_status("cancelled", "Model loading was cancelled.", 0)
        else:
            logger.error(f"Model loading error: {e}", exc_info=True)
            _update_download_status("error", "", 0, str(e))
        MODEL_LOADED = False
        return False
    except ImportError as e:
        logger.critical(f"Missing package: {e}", exc_info=True)
        _update_download_status("error", "", 0, str(e))
        MODEL_LOADED = False
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        _update_download_status("error", "", 0, str(e))
        dia_model = None
        model_config_instance = None
        MODEL_LOADED = False
        return False


def _load_dia1_model(repo_id: str, cache_path: str):
    """Loads a Dia 1 (original) model."""
    global dia_model, model_config_instance, model_device, EXPECTED_SAMPLE_RATE

    config_filename = get_model_config_filename()
    weights_filename = get_model_weights_filename()
    compute_dtype_str = get_compute_dtype(model_device, weights_filename)

    logger.info(f"  Config file: {config_filename}")
    logger.info(f"  Weights file: {weights_filename}")
    logger.info(f"  Compute dtype: {compute_dtype_str}")

    # Phase 1: Download config
    _check_cancelled()
    _update_download_status("downloading", f"Downloading config from {repo_id}...", 10)

    local_config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        cache_dir=cache_path,
    )
    logger.info(f"Configuration file path: {local_config_path}")

    # Phase 2: Download weights
    _check_cancelled()
    _update_download_status("downloading", f"Downloading weights ({weights_filename})...", 30)

    local_weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_filename,
        cache_dir=cache_path,
    )
    logger.info(f"Weights file path: {local_weights_path}")

    # Phase 3: Load model
    _check_cancelled()
    _update_download_status("loading", "Loading Dia 1 model...", 60)

    config = DiaConfig.load(local_config_path)
    if config is None:
        raise FileNotFoundError(f"Config file failed to load from {local_config_path}")

    dia_instance = Dia1Model(config, compute_dtype=compute_dtype_str, device=model_device)

    # Load weights
    logger.info(f"Loading weights from: {local_weights_path}")
    map_location = torch.device("cpu")
    if local_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(local_weights_path, device=str(map_location))
    elif local_weights_path.endswith(".pth"):
        state_dict = torch.load(local_weights_path, map_location=map_location)
    else:
        raise ValueError(f"Unsupported weights file format: {weights_filename}")

    _check_cancelled()
    _update_download_status("loading", "Applying weights...", 75)

    dia_instance.model.load_state_dict(state_dict)
    dia_instance.model.to(model_device)
    dia_instance.model.eval()

    # Phase 4: Load DAC model
    _check_cancelled()
    _update_download_status("loading", "Loading DAC audio codec...", 85)

    dia_instance._load_dac_model()

    dia_model = dia_instance
    model_config_instance = dia_model.config
    EXPECTED_SAMPLE_RATE = DIA1_SAMPLE_RATE

    _update_download_status("loading", "Dia 1 model ready.", 95)


def _load_dia2_model(repo_id: str, cache_path: str):
    """Loads a Dia 2 model using Dia2.from_repo()."""
    global dia_model, model_config_instance, model_device, EXPECTED_SAMPLE_RATE

    device_str = str(model_device) if model_device else "cuda"

    # Determine dtype based on device
    if model_device and model_device.type == "cuda":
        dtype_str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    elif model_device and model_device.type == "mps":
        dtype_str = "float16"
    else:
        dtype_str = "float32"

    logger.info(f"  Dia 2 dtype: {dtype_str}")

    # Phase 1: Download and load (Dia2.from_repo handles this)
    _check_cancelled()
    _update_download_status("downloading", f"Downloading {repo_id}...", 20)

    # Dia2.from_repo downloads from HF and loads the model
    _check_cancelled()
    _update_download_status("loading", f"Loading Dia 2 model ({repo_id})...", 50)

    dia2_instance = Dia2.from_repo(
        repo_id,
        device=device_str,
        dtype=dtype_str,
    )

    _check_cancelled()
    _update_download_status("loading", "Dia 2 model ready.", 95)

    dia_model = dia2_instance
    model_config_instance = None  # Dia 2 doesn't expose config the same way
    EXPECTED_SAMPLE_RATE = dia2_instance.sample_rate

    logger.info(f"Dia 2 sample rate: {EXPECTED_SAMPLE_RATE}")


def unload_model() -> bool:
    """
    Unloads the current model and releases resources.
    Does NOT reload - use reload_model_async() for hot-swap.
    """
    global dia_model, model_config_instance, MODEL_LOADED
    global loaded_model_selector, loaded_model_type, model_device

    logger.info("Initiating model unload sequence...")

    if dia_model is not None:
        # Try to close Dia2 model gracefully
        if loaded_model_type == "dia2" and hasattr(dia_model, "close"):
            try:
                dia_model.close()
            except Exception as e:
                logger.warning(f"Error closing Dia2 model: {e}")

        del dia_model
        dia_model = None

    if model_config_instance is not None:
        del model_config_instance
        model_config_instance = None

    MODEL_LOADED = False
    loaded_model_selector = None
    loaded_model_type = None

    # Force garbage collection and clear GPU cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared after model unload.")

    logger.info("Model unloaded successfully.")
    return True


def _reload_model_worker():
    """
    Background worker that performs the actual model reload.
    Runs in a separate thread so the FastAPI server stays responsive.
    """
    try:
        _update_download_status("unloading", "Unloading current model...", 5)
        unload_model()

        _check_cancelled()
        logger.info("Resources cleared. Loading model from updated config...")
        load_model()
    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Reload worker cancelled: {e}")
            _update_download_status("cancelled", "Model loading was cancelled.", 0)
        else:
            logger.error(f"Reload worker error: {e}", exc_info=True)
            _update_download_status("error", "", 0, str(e))
    except Exception as e:
        logger.error(f"Reload worker error: {e}", exc_info=True)
        _update_download_status("error", "", 0, str(e))


def reload_model_async():
    """
    Initiates a model hot-swap in a background thread.
    Returns immediately so the server can continue serving status polls.
    If a load is already in progress, cancels it first.
    """
    global _load_thread

    logger.info("Initiating async model hot-swap/reload sequence...")

    # Cancel any in-progress load
    if is_loading():
        logger.info("Cancelling in-progress model load...")
        _cancel_event.set()
        _load_thread.join(timeout=15)
        logger.info("Previous load thread finished.")

    # Reset cancel flag for new load
    _cancel_event.clear()

    # Start new background load
    _load_thread = threading.Thread(target=_reload_model_worker, daemon=True)
    _load_thread.start()
    logger.info("Background model reload thread started.")


def cancel_loading():
    """Cancels any in-progress model loading."""
    if is_loading():
        logger.info("Cancelling model loading by user request...")
        _cancel_event.set()
        _update_download_status("cancelling", "Cancelling model load...", 0)
        return True
    return False


def reload_model() -> bool:
    """
    Synchronous reload (used for startup only).
    Unloads current model, clears resources, reloads from config.
    """
    logger.info("Initiating synchronous model reload...")
    _cancel_event.clear()
    _update_download_status("unloading", "Unloading current model...", 5)
    unload_model()
    logger.info("Resources cleared. Loading model from updated config...")
    return load_model()


# --- Cloning Preparation Helper ---


def _prepare_cloning_inputs(
    clone_reference_filename: str,
    reference_audio_base_path: str,
    max_ref_duration_sec: float,
    whisper_model_name: str,
    whisper_cache_path: str,
    transcript: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:  # MODIFIED return type
    """
    Prepares inputs for voice cloning: loads/processes audio, gets transcript.

    Args:
        clone_reference_filename: Filename of the reference audio.
        reference_audio_base_path: Base path where reference files are stored.
        max_ref_duration_sec: Maximum duration for the reference audio prompt.
        whisper_model_name: Name of the Whisper model to use if needed.
        whisper_cache_path: Path to Whisper model cache.
        transcript: Optional explicit transcript text to override file/Whisper.

    Returns:
        Tuple of (audio_prompt_tensor, reference_transcript_text, error_message).
        On success, error_message is None. On failure, text and tensor are None.
    """
    global dia_model  # Need access to the loaded Dia model for DAC

    reference_audio_path = os.path.join(
        reference_audio_base_path, clone_reference_filename
    )
    if not os.path.isfile(reference_audio_path):
        return None, None, f"Reference audio file not found: {reference_audio_path}"

    # --- 1. Load and Process Audio (CPU only) ---
    try:
        logger.info(f"Loading reference audio: {reference_audio_path}")
        # Load on CPU without moving to device
        audio_tensor, sr = torchaudio.load(reference_audio_path)

        # Ensure correct sample rate
        if sr != EXPECTED_SAMPLE_RATE:
            logger.warning(
                f"Reference audio SR ({sr}Hz) differs from expected ({EXPECTED_SAMPLE_RATE}Hz). Resampling..."
            )
            resampled_tensor = torchaudio.functional.resample(
                audio_tensor, sr, EXPECTED_SAMPLE_RATE
            )
            del audio_tensor  # Free the original tensor
            audio_tensor = resampled_tensor

        # Ensure mono
        if audio_tensor.shape[0] > 1:
            logger.warning(
                f"Reference audio '{clone_reference_filename}' is stereo. Converting to mono."
            )
            mono_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            del audio_tensor  # Free the stereo tensor
            audio_tensor = mono_tensor

        # Truncate if necessary
        num_samples = audio_tensor.shape[1]
        duration_sec = num_samples / EXPECTED_SAMPLE_RATE

        if duration_sec > max_ref_duration_sec:
            logger.warning(
                f"Reference audio duration ({duration_sec:.2f}s) exceeds max ({max_ref_duration_sec:.2f}s). Truncating..."
            )
            target_samples = int(max_ref_duration_sec * EXPECTED_SAMPLE_RATE)
            truncated_tensor = audio_tensor[:, :target_samples].clone()
            del audio_tensor  # Free the full audio tensor
            audio_tensor = truncated_tensor
            new_duration = audio_tensor.shape[1] / EXPECTED_SAMPLE_RATE
            logger.info(f"Truncated reference audio to {new_duration:.2f}s.")
        else:
            logger.info(
                f"Reference audio duration ({duration_sec:.2f}s) is within limit."
            )

        # Convert processed tensor to NumPy array (float32) for potential Whisper use
        processed_audio_np = audio_tensor.squeeze(0).numpy().astype(np.float32)

        # Clear tensors when done with them
        del audio_tensor

    except Exception as e:
        logger.error(
            f"Error loading/processing reference audio '{reference_audio_path}': {e}",
            exc_info=True,
        )
        return None, None, f"Failed to load/process reference audio: {e}"

    # --- 2. Get Transcript ---
    transcript_text: Optional[str] = None
    error_message: Optional[str] = None
    transcript_source: str = "unknown"

    if transcript is not None:
        logger.info("Using provided transcript override for cloning.")
        transcript_text = transcript.strip()
        transcript_source = "explicit"
        # Check and prepend [S1] or [S2] if needed (assuming clone target is usually S1)
        if not transcript_text.startswith(("[S1]", "[S2]")):
            logger.debug("Prepending '[S1] ' to explicit transcript.")
            transcript_text = "[S1] " + transcript_text

    else:
        # Try loading local .txt file
        base_name, _ = os.path.splitext(clone_reference_filename)
        transcript_filename = base_name + ".txt"
        transcript_filepath = os.path.join(
            reference_audio_base_path, transcript_filename
        )
        logger.info(f"Checking for local transcript: {transcript_filepath}")

        if os.path.isfile(transcript_filepath):
            try:
                with open(transcript_filepath, "r", encoding="utf-8") as f:
                    transcript_text = f.read().strip()
                logger.info(f"Loaded transcript from local file: {transcript_filepath}")
                transcript_source = "file"
                # Assume file is correctly formatted (includes speaker tags)
                # Ensure tag exists just in case file is malformed
                if not transcript_text.startswith(("[S1]", "[S2]")):
                    logger.warning(
                        f"Local transcript file '{transcript_filepath}' missing speaker tag. Prepending '[S1]'."
                    )
                    transcript_text = "[S1] " + transcript_text
            except Exception as e:
                logger.warning(
                    f"Failed to read local transcript file '{transcript_filepath}': {e}. Will attempt Whisper.",
                    exc_info=True,
                )
                transcript_text = None  # Ensure it's None so Whisper runs

        if transcript_text is None:
            # Try Whisper
            logger.info(
                "Local transcript not found or failed to load. Attempting Whisper generation..."
            )
            generated_transcript = _generate_transcript_with_whisper(
                processed_audio_np, whisper_model_name, whisper_cache_path
            )

            if generated_transcript is not None:
                transcript_text = "[S1] " + generated_transcript.strip()  # Prepend [S1]
                transcript_source = "whisper"
                logger.info("Whisper transcription successful.")
                # Save the generated transcript
                try:
                    with open(transcript_filepath, "w", encoding="utf-8") as f:
                        f.write(transcript_text)  # Save with the [S1] tag
                    logger.info(f"Saved Whisper transcript to: {transcript_filepath}")
                except Exception as e:
                    logger.warning(
                        f"Failed to save generated transcript to '{transcript_filepath}': {e}",
                        exc_info=True,
                    )
            else:
                logger.error("Whisper transcription failed.")
                error_message = "Reference transcript file not found and automatic transcription failed."
                transcript_source = "failed"

    # --- 3. Check if Transcript was Obtained ---
    if transcript_text is None:
        # Free resources before returning
        del processed_audio_np
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, error_message or "Failed to obtain reference transcript."

    # Free resources before returning
    del processed_audio_np
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return without audio_prompt_tensor (step 4 deleted)
    return None, transcript_text, None


# --- Helper Function for Robust File Finding ---


def _find_reference_file(filename_input: str, base_path: str) -> Optional[str]:
    """
    Finds a reference audio file (.wav or .mp3) in the base_path,
    handling case-insensitivity, missing extensions, and extra whitespace/paths.

    Args:
        filename_input: The potentially "dirty" filename provided by the user/API.
        base_path: The directory path where reference files are stored.

    Returns:
        The actual filename (with correct casing) as it exists on the filesystem,
        or None if no suitable match is found.
    """
    if not filename_input or not base_path or not os.path.isdir(base_path):
        logger.warning(
            f"_find_reference_file: Invalid input filename ('{filename_input}') or base_path ('{base_path}')."
        )
        return None

    # 1. Normalize input: remove path components and whitespace
    target_basename = os.path.basename(filename_input).strip()
    if not target_basename:
        logger.warning(
            f"_find_reference_file: Input '{filename_input}' resulted in empty basename after normalization."
        )
        return None

    # 2. Get the base name (without extension) and lowercase it for comparison
    target_base, target_ext = os.path.splitext(target_basename)
    target_base_lower = target_base.lower()

    logger.debug(
        f"_find_reference_file: Normalized input to base='{target_base_lower}', original_ext='{target_ext}'"
    )

    # 3. List actual files and compare case-insensitively
    try:
        for actual_filename in os.listdir(base_path):
            if os.path.isfile(os.path.join(base_path, actual_filename)):
                actual_base, actual_ext = os.path.splitext(actual_filename)
                actual_base_lower = actual_base.lower()
                actual_ext_lower = actual_ext.lower()

                # Check if base names match (case-insensitive)
                if actual_base_lower == target_base_lower:
                    # Check if the actual extension is valid (.wav or .mp3)
                    if actual_ext_lower in [".wav", ".mp3"]:
                        logger.info(
                            f"_find_reference_file: Found match for '{filename_input}' -> '{actual_filename}'"
                        )
                        return actual_filename  # Return the filename with its original casing

    except OSError as e:
        logger.error(
            f"Error listing reference directory '{base_path}': {e}", exc_info=True
        )
        return None  # Indicate failure to list directory

    logger.warning(
        f"_find_reference_file: No suitable match found for '{filename_input}' in '{base_path}'."
    )
    return None  # No match found


# --- Dia 2 Speech Generation ---


def _generate_speech_dia2(
    text_to_process: str,
    voice_mode: str = "single_s1",
    clone_reference_filename: Optional[str] = None,
    cfg_scale: float = 2.0,
    temperature: float = 0.8,
    top_p: float = 0.95,
    speed_factor: float = 1.0,
    cfg_filter_top_k: int = 50,
    seed: int = -1,
    split_text: bool = False,
    chunk_size: int = 120,
    enable_silence_trimming: bool = True,
    enable_internal_silence_fix: bool = True,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Generates speech using a loaded Dia 2 model.
    Uses GenerationConfig and prefix_speaker for voice conditioning.
    """
    global dia_model

    if dia_model is None:
        logger.error("Dia 2 model is not loaded.")
        return None

    monitor = PerformanceMonitor()
    monitor.record("Request received in engine (Dia 2)")

    # Seeding
    if seed >= 0:
        logger.info(f"Using generation seed: {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Build GenerationConfig
    gen_config = GenerationConfig(
        cfg_scale=cfg_scale,
        cfg_filter_k=cfg_filter_top_k,
        audio=SamplingConfig(temperature=temperature, top_k=cfg_filter_top_k),
    )

    # Determine prefix speakers for cloning
    prefix_s1 = None
    prefix_s2 = None
    if voice_mode == "clone" and clone_reference_filename:
        # For Dia 2, we use prefix_speaker_1 with the reference audio path
        prefix_s1 = clone_reference_filename
        logger.info(f"Dia 2 cloning: using prefix_speaker_1={clone_reference_filename}")

    # Text splitting
    if split_text and len(text_to_process) >= chunk_size * 2:
        text_chunks = chunk_text_by_sentences(text_to_process, chunk_size)
        logger.info(f"Split text into {len(text_chunks)} chunks.")
    else:
        text_chunks = [text_to_process] if text_to_process.strip() else []

    if not text_chunks:
        logger.warning("No text chunks to process.")
        return None

    all_audio_arrays: List[np.ndarray] = []
    total_chunks = len(text_chunks)
    logger.info(f"Starting Dia 2 generation for {total_chunks} chunks.")

    outer_pbar = None
    if total_chunks > 1:
        outer_pbar = tqdm(total=total_chunks, desc="Dia 2 Chunks", unit="chunk", position=0, leave=True)

    try:
        for i, chunk in enumerate(text_chunks):
            chunk_start_time = time.time()
            if outer_pbar:
                outer_pbar.set_description(f"Chunk {i+1}/{total_chunks}")

            logger.info(f"Processing chunk {i+1}/{total_chunks} with Dia 2 generate()...")

            try:
                result = dia_model.generate(
                    chunk,
                    config=gen_config,
                    prefix_speaker_1=prefix_s1,
                    prefix_speaker_2=prefix_s2,
                    include_prefix=False,
                    verbose=True,
                )

                # GenerationResult has .waveform (torch.Tensor) and .sample_rate
                if result is not None and result.waveform is not None:
                    waveform = result.waveform
                    if isinstance(waveform, torch.Tensor):
                        chunk_audio = waveform.cpu().numpy().squeeze()
                    else:
                        chunk_audio = np.array(waveform).squeeze()

                    if chunk_audio.size > 0:
                        all_audio_arrays.append(chunk_audio.astype(np.float32))
                        chunk_duration = time.time() - chunk_start_time
                        logger.info(f"Chunk {i+1} generated in {chunk_duration:.2f}s. Shape: {chunk_audio.shape}")
                    else:
                        logger.warning(f"Dia 2 generate() returned empty audio for chunk {i+1}.")
                else:
                    logger.warning(f"Dia 2 generate() returned None for chunk {i+1}.")

            except Exception as gen_exc:
                logger.error(f"Error in Dia 2 generate() for chunk {i+1}: {gen_exc}", exc_info=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if outer_pbar:
                outer_pbar.update(1)

        if outer_pbar:
            outer_pbar.close()

        if not all_audio_arrays:
            logger.error("No audio generated from any chunk using Dia 2.")
            return None

        final_audio_np = np.concatenate(all_audio_arrays)
        all_audio_arrays.clear()
        logger.info(f"Concatenated Dia 2 audio shape: {final_audio_np.shape}")

    except Exception as e:
        if outer_pbar and not outer_pbar.disable:
            outer_pbar.close()
        logger.error(f"Error during Dia 2 generation loop: {e}", exc_info=True)
        return None

    # Apply speed factor
    if speed_factor != 1.0:
        logger.info(f"Applying speed factor: {speed_factor}")
        original_len = len(final_audio_np)
        speed_factor = max(0.5, min(speed_factor, 2.0))
        target_len = int(original_len / speed_factor)
        if target_len > 0 and target_len != original_len:
            x_original = np.linspace(0, 1, original_len)
            x_resampled = np.linspace(0, 1, target_len)
            final_audio_np = np.interp(x_resampled, x_original, final_audio_np).astype(np.float32)

    final_audio_np = final_audio_np.astype(np.float32)

    # Post-processing
    if enable_silence_trimming:
        final_audio_np = trim_lead_trail_silence(final_audio_np, sample_rate=EXPECTED_SAMPLE_RATE)
    if enable_internal_silence_fix:
        final_audio_np = fix_internal_silence(final_audio_np, sample_rate=EXPECTED_SAMPLE_RATE)

    logger.info(f"Dia 2 final audio shape: {final_audio_np.shape}")
    logger.debug(monitor.report())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_audio_np, EXPECTED_SAMPLE_RATE


# --- Speech Generation (Unified Dispatch) ---


def generate_speech(
    text_to_process: str,
    voice_mode: str = "single_s1",
    clone_reference_filename: Optional[str] = None,
    transcript: Optional[str] = None,  # Explicit transcript for cloning prep
    max_tokens: Optional[
        int
    ] = None,  # Kept for signature compatibility, may not be used by model.generate
    cfg_scale: float = 3.0,
    temperature: float = 1.3,
    top_p: float = 0.95,
    speed_factor: float = 0.94,  # Applied post-generation
    cfg_filter_top_k: int = 35,
    seed: int = 42,
    split_text: bool = False,
    chunk_size: int = 120,
    # Post-processing parameters (applied after generation)
    enable_silence_trimming: bool = True,
    enable_internal_silence_fix: bool = True,
    enable_unvoiced_removal: bool = True,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Unified speech generation dispatcher.
    Routes to the appropriate backend based on the loaded model type (Dia 1 or Dia 2).
    """
    global dia_model

    if not MODEL_LOADED or dia_model is None or not model_device:
        logger.error("No model loaded. Cannot generate speech.")
        return None

    # --- Dispatch to Dia 2 backend ---
    if loaded_model_type == "dia2":
        return _generate_speech_dia2(
            text_to_process=text_to_process,
            voice_mode=voice_mode,
            clone_reference_filename=clone_reference_filename,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            speed_factor=speed_factor,
            cfg_filter_top_k=cfg_filter_top_k,
            seed=seed,
            split_text=split_text,
            chunk_size=chunk_size,
            enable_silence_trimming=enable_silence_trimming,
            enable_internal_silence_fix=enable_internal_silence_fix,
        )

    # --- Dia 1 (original) backend below ---

    # Clear CUDA cache before starting new generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    monitor = PerformanceMonitor()
    monitor.record("Request received in engine (Dia 1 generate)")

    # Basic split_text logic (same as original)
    if split_text:
        if len(text_to_process) < chunk_size * 2:
            split_text = False

    # Logging (same as original)
    log_params = {
        "mode": voice_mode,
        "seed": seed,
        "split": split_text,
        "chunk_size": chunk_size,
        "max_tokens": max_tokens if max_tokens is not None else "ModelDefault",
        "cfg": cfg_scale,
        "temp": temperature,
        "top_p": top_p,
        "top_k": cfg_filter_top_k,
        "speed": speed_factor,
        "clone_ref": clone_reference_filename if clone_reference_filename else "N/A",
        "transcript_provided": transcript is not None,
        "text_snippet": f"'{text_to_process[:80].replace(os.linesep, ' ')}...'",
    }
    logger.info(f"Generating speech (simple method) with params: {log_params}")

    # --- Seeding (same as original) ---
    if seed >= 0:
        logger.info(f"Using generation seed: {seed}")
        torch.manual_seed(seed)
        if model_device.type == "cuda":
            logger.debug("Applying CUDA seed.")
            torch.cuda.manual_seed_all(seed)
    else:
        logger.info("Using random generation seed.")

    # --- Prepare Cloning Info (if needed) ---
    reference_transcript_text: Optional[str] = None
    validated_clone_audio_path: Optional[str] = None  # Store the full path

    if voice_mode == "clone":
        if not clone_reference_filename:
            logger.error("Clone mode selected but no reference filename provided.")
            return None

        # Check if clone_reference_filename is already a full path to an existing file
        reference_base_path = get_reference_audio_path()
        if os.path.isfile(clone_reference_filename):
            logger.info(
                f"Using provided full path for cloning: {clone_reference_filename}"
            )
            validated_clone_audio_path = clone_reference_filename
        else:
            # Traditional behavior: treat as filename in reference_audio directory
            validated_filename = _find_reference_file(  # Get validated filename
                clone_reference_filename, reference_base_path
            )

            if not validated_filename:
                logger.error(
                    f"Could not find a valid reference file matching '{clone_reference_filename}' in '{reference_base_path}'."
                )
                return None

            validated_clone_audio_path = os.path.join(
                reference_base_path, validated_filename
            )  # Store full path

        # Call _prepare_cloning_inputs primarily to get the transcript text
        # We ignore the returned audio_prompt_tensor as model.generate takes the path
        _, ref_transcript, prep_error = _prepare_cloning_inputs(
            clone_reference_filename=validated_clone_audio_path,  # Use validated name
            reference_audio_base_path=reference_base_path,
            max_ref_duration_sec=20.0,  # This duration limit still applies to the audio loaded internally by model.generate
            whisper_model_name=get_whisper_model_name(),
            whisper_cache_path=get_model_cache_path(),
            transcript=transcript,  # Pass optional explicit transcript
        )

        if prep_error:
            logger.error(
                f"Cloning preparation failed (getting transcript): {prep_error}"
            )
            # If transcript fails, we cannot replicate the author's method, so fail.
            return None

        reference_transcript_text = ref_transcript
        logger.info(
            f"Prepared for cloning (simple method): using audio path '{validated_clone_audio_path}', transcript_len={len(reference_transcript_text) if reference_transcript_text else 0}"
        )

        # Reset model state after cloning preparation
        if hasattr(dia_model, "reset_state"):
            dia_model.reset_state()
            logger.debug("Model state reset after cloning preparation")

    monitor.record("Parameters processed (simple generate)")

    # --- Text Splitting (same as original) ---
    text_chunks: List[str] = []
    if split_text:
        logger.info(
            f"Splitting text into sentence-based chunks (max aggregation size: {chunk_size})..."
        )
        text_chunks = chunk_text_by_sentences(text_to_process, chunk_size)
        # ... (handle empty chunking result - same as original) ...
        logger.info(f"Split text into {len(text_chunks)} chunks.")
    else:
        logger.info("Text splitting disabled. Processing text as a single chunk.")
        if text_to_process.strip():
            text_chunks.append(text_to_process)

    if not text_chunks:
        logger.warning("No text chunks to process after splitting.")
        if hasattr(dia_model, "reset_state"):
            dia_model.reset_state()
        return None  # Nothing to generate

    # --- Generation Section (NEW LOGIC) ---
    all_audio_arrays: List[np.ndarray] = []
    total_chunks = len(text_chunks)
    logger.info(
        f"Starting generation loop for {total_chunks} chunks using model.generate() per chunk."
    )

    # Progress bar setup (optional, similar to original)
    show_outer_pbar = total_chunks > 1
    outer_pbar = None
    if show_outer_pbar:
        outer_pbar = tqdm(
            total=total_chunks,
            desc="Processing Chunks (Simple)",
            unit="chunk",
            position=0,
            leave=True,
        )  # Leave bar after completion

    try:
        for i, chunk in enumerate(text_chunks):
            chunk_start_time = time.time()
            if outer_pbar:
                outer_pbar.set_description(f"Chunk {i+1}/{total_chunks} (Simple)")

            logger.info(
                f"Processing chunk {i+1}/{total_chunks} with model.generate()..."
            )

            # Determine inputs for model.generate based on mode
            text_input_for_generate: str
            audio_prompt_for_generate: Optional[str] = None  # Expects path or None

            if (
                voice_mode == "clone"
                and reference_transcript_text
                and validated_clone_audio_path
            ):
                # Prepend transcript, pass audio path
                text_input_for_generate = reference_transcript_text + " " + chunk
                # if i == 0:
                #     text_input_for_generate = reference_transcript_text + " " + chunk
                # else:
                #     text_input_for_generate = chunk

                audio_prompt_for_generate = validated_clone_audio_path
                logger.debug(
                    f"Chunk {i+1} (Clone): Using audio prompt path '{audio_prompt_for_generate}' and prepended text."
                )
            else:
                # Non-cloning mode: Use only the chunk text
                text_input_for_generate = chunk
                audio_prompt_for_generate = None
                logger.debug(f"Chunk {i+1} (Non-Clone): Using text chunk directly.")

            # Call the model's generate function for this chunk
            try:
                # Note: We assume dia_model is the preloaded instance of the Dia class
                chunk_output_np = dia_model.generate(
                    text=text_input_for_generate,
                    audio_prompt=audio_prompt_for_generate,  # Pass audio file path or None
                    # Pass generation parameters directly
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                    # max_tokens=max_tokens, # model.generate might not support this limit directly
                    use_torch_compile=False,  # Set as needed, False for simplicity
                    verbose=True,  # Set verbose=True for detailed logs from model.generate per chunk
                    text_to_generate_size=len(chunk),
                    seed=seed,
                )
                monitor.record(f"model.generate completed for chunk {i+1}")

                if chunk_output_np is not None and chunk_output_np.size > 0:
                    all_audio_arrays.append(chunk_output_np)
                    chunk_duration = time.time() - chunk_start_time
                    logger.info(
                        f"Chunk {i+1} generated successfully in {chunk_duration:.2f}s. Audio shape: {chunk_output_np.shape}"
                    )
                    # Clear CUDA cache after each chunk
                    torch.cuda.empty_cache()
                    # Reset model state after each chunk to free memory
                    if hasattr(dia_model, "reset_state"):
                        dia_model.reset_state()
                else:
                    logger.warning(
                        f"model.generate() returned None or empty audio for chunk {i+1}. Skipping."
                    )

            except Exception as gen_exc:
                logger.error(
                    f"Error calling model.generate() for chunk {i+1}: {gen_exc}",
                    exc_info=True,
                )
                # Clean up after generation error
                if hasattr(dia_model, "reset_state"):
                    dia_model.reset_state()
                # Option: Stop processing further chunks on error, or just skip this one
                # For robustness, let's skip and continue, but log the error.
                # raise # Or re-raise to stop the whole process

            if outer_pbar:
                outer_pbar.update(1)

        # --- End of chunk loop ---
        if outer_pbar:
            outer_pbar.close()

        # --- Concatenate Audio Chunks ---
        if not all_audio_arrays:
            logger.error("No audio generated from any chunk using the simple method.")
            if hasattr(dia_model, "reset_state"):
                dia_model.reset_state()
            final_audio_np = None
        else:
            monitor.record("Concatenating audio chunks (simple generate)")
            final_audio_np = np.concatenate(all_audio_arrays)
            monitor.record("Concatenation complete (simple generate)")
            logger.info(
                f"Concatenated audio shape (simple method): {final_audio_np.shape}"
            )
            # Clear array and force collection
            all_audio_arrays.clear()  # Remove references to chunk arrays
            torch.cuda.empty_cache()  # Clear CUDA cache

    except Exception as e:
        if outer_pbar and not outer_pbar.disable:
            outer_pbar.close()
        logger.error(f"Error during simplified generation loop: {e}", exc_info=True)
        logger.debug(monitor.report())
        final_audio_np = None  # Indicate failure
        # Clean up in error case
        if "all_audio_arrays" in locals() and all_audio_arrays:
            all_audio_arrays.clear()
        if hasattr(dia_model, "reset_state"):
            dia_model.reset_state()
        torch.cuda.empty_cache()

    # Check if generation failed
    if final_audio_np is None:
        logger.error("Audio generation process failed (simple method).")
        return None

    # --- Apply Post-processing (Speed Factor, Silence Trim, etc. - Same as original) ---
    # Apply Speed Factor
    if speed_factor != 1.0:
        # ... (same speed factor logic as in original generate_speech) ...
        logger.info(f"Applying speed factor: {speed_factor}")
        monitor.record("Applying speed factor")
        original_len = len(final_audio_np)
        speed_factor = max(0.5, min(speed_factor, 2.0))
        target_len = int(original_len / speed_factor)
        if target_len > 0 and target_len != original_len:
            x_original = np.linspace(0, 1, original_len)
            x_resampled = np.linspace(0, 1, target_len)
            resampled_audio_np = np.interp(x_resampled, x_original, final_audio_np)
            final_audio_np = resampled_audio_np.astype(np.float32)
            monitor.record("Speed factor applied")
            logger.info(
                f"Audio resampled for {speed_factor:.2f}x speed. New shape: {final_audio_np.shape}"
            )
            # Delete the original and clear cache
            del x_original, x_resampled, resampled_audio_np
            torch.cuda.empty_cache()
        else:
            logger.warning(
                f"Speed factor {speed_factor} resulted in invalid target length {target_len} or no change needed. Skipping speed adjustment."
            )
            final_audio_np = final_audio_np.astype(np.float32)  # Ensure correct type
    else:
        logger.info("Speed factor is 1.0, no speed adjustment needed.")
        final_audio_np = final_audio_np.astype(np.float32)  # Ensure correct type

    # Apply other post-processing to the *final concatenated* audio
    audio_changes = {"trim": False, "silence": False, "unvoiced": False}
    original_final_length = len(final_audio_np)
    logger.info("Applying final audio post-processing...")

    if enable_silence_trimming:
        # ... (call trim_lead_trail_silence) ...
        len_before = len(final_audio_np)
        final_audio_np = trim_lead_trail_silence(
            final_audio_np, sample_rate=EXPECTED_SAMPLE_RATE
        )
        if len(final_audio_np) != len_before:
            audio_changes["trim"] = True

    if enable_internal_silence_fix:
        # ... (call fix_internal_silence) ...
        len_before = len(final_audio_np)
        final_audio_np = fix_internal_silence(
            final_audio_np, sample_rate=EXPECTED_SAMPLE_RATE
        )
        if len(final_audio_np) != len_before:
            audio_changes["silence"] = True

    # if enable_unvoiced_removal:
    #     # ... (call remove_long_unvoiced_segments) ...
    #     len_before = len(final_audio_np)
    #     final_audio_np = remove_long_unvoiced_segments(
    #         final_audio_np, sample_rate=EXPECTED_SAMPLE_RATE
    #     )
    #     if len(final_audio_np) != len_before:
    #         audio_changes["unvoiced"] = True

    # Log post-processing summary
    new_final_length = len(final_audio_np)
    if any(audio_changes.values()):
        change_percent = (
            ((new_final_length - original_final_length) / original_final_length * 100)
            if original_final_length > 0
            else 0
        )
        applied_steps = ", ".join(k for k, v in audio_changes.items() if v)
        logger.info(
            f"  → Final Post-processing applied ({applied_steps}): {original_final_length} → {new_final_length} samples ({change_percent:.1f}% change)"
        )
    else:
        logger.info("  → No significant changes from final audio post-processing")

    # Final checks (same as original)
    if final_audio_np.dtype != np.float32:
        logger.warning(
            f"Final audio was not float32 ({final_audio_np.dtype}), converting."
        )
        final_audio_np = final_audio_np.astype(np.float32)

    logger.info(
        f"Final audio ready (simple method). Shape: {final_audio_np.shape}, dtype: {final_audio_np.dtype}"
    )
    logger.debug(monitor.report())

    # Final memory cleanup
    torch.cuda.empty_cache()
    if hasattr(dia_model, "reset_state"):
        dia_model.reset_state()

    return final_audio_np, EXPECTED_SAMPLE_RATE
