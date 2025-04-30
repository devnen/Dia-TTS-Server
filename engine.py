# engine.py
# Core Dia TTS model loading and generation logic, adapted for new dia library structure

import logging
import time
import os
import torch
import torchaudio  # Import torchaudio for loading/processing
import numpy as np
from typing import Optional, Tuple, List, Dict, Any  # Added Dict, Any
from huggingface_hub import hf_hub_download
from tqdm import tqdm  # Import tqdm for progress bars

# Import Dia model class and config from the NEW dia library structure
try:
    from dia.model import (
        Dia,
        ComputeDtype,
        DEFAULT_SAMPLE_RATE,
    )  # Use new Dia class and constants
    from dia.config import DiaConfig

    # Import state management classes
    from dia.state import EncoderInferenceState, DecoderInferenceState, DecoderOutput
except ImportError as e:
    logging.critical(
        f"Failed to import NEW Dia model components: {e}. Ensure the updated 'dia' package exists and is importable.",
        exc_info=True,
    )

    # Define dummy classes/functions to prevent server crash on import,
    # but generation will fail later if these are used.
    class Dia:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_local(*args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def generate(*args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def _prepare_text_input(self, *args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def load_audio(self, *args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def _prepare_audio_prompt(self, *args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def _decoder_step(self, *args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def _generate_output(self, *args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        # Add dummy _load_dac_model if needed by other parts
        def _load_dac_model(self, *args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

    class DiaConfig:
        pass

    class EncoderInferenceState:
        @staticmethod
        def new(*args, **kwargs):
            pass

    class DecoderInferenceState:
        @staticmethod
        def new(*args, **kwargs):
            pass

    class DecoderOutput:
        @staticmethod
        def new(*args, **kwargs):
            pass

    ComputeDtype = None
    DEFAULT_SAMPLE_RATE = 44100  # Fallback sample rate


# Import configuration getters from our project's config.py
from config import (
    get_model_repo_id,
    get_model_cache_path,
    get_reference_audio_path,
    get_model_config_filename,
    get_model_weights_filename,
    get_gen_default_seed,  # Import seed getter
    get_whisper_model_name,  # Import Whisper config getter
)

# Import text splitting utility and other helpers
from utils import (
    chunk_text_by_sentences,
    PerformanceMonitor,
    trim_lead_trail_silence,
    fix_internal_silence,
    remove_long_unvoiced_segments,
    _generate_transcript_with_whisper,  # Import Whisper helper
)

logger = logging.getLogger(__name__)

# --- Global Variables ---
dia_model: Optional[Dia] = None
model_config_instance: Optional[DiaConfig] = None  # Keep for potential reference
model_device: Optional[torch.device] = None
MODEL_LOADED = False
# Use sample rate from the imported dia library if available
EXPECTED_SAMPLE_RATE = DEFAULT_SAMPLE_RATE

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


def get_compute_dtype(device: torch.device) -> str:  # Changed type hint to str
    """Determines the recommended compute dtype string based on device capabilities."""
    # Ensure ComputeDtype was imported correctly before using it for comparison logic
    # We still need the Enum for the logic, but return the string value.
    if ComputeDtype is None:
        logger.warning(
            "ComputeDtype enum not available from dia.model, defaulting to 'float32' string."
        )
        return "float32"  # Return string directly

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            logger.info("CUDA device supports bfloat16, using BF16 compute dtype.")
            return ComputeDtype.BFLOAT16.value  # Return string value
        else:
            logger.info(
                "CUDA device does not support bfloat16, using FP16 compute dtype."
            )
            return ComputeDtype.FLOAT16.value  # Return string value
    elif device.type == "mps":
        logger.info("MPS device detected, using FP16 compute dtype.")
        # Note: FP16 might be the only practical option on MPS for many models
        return ComputeDtype.FLOAT16.value  # Return string value
    else:  # CPU
        logger.info("CPU device detected, using FP32 compute dtype.")
        return ComputeDtype.FLOAT32.value  # Return string value


def load_model():
    """
    Loads the Dia TTS model and associated DAC model using the new Dia class methods.
    Downloads model files based on configuration if they don't exist locally.
    Handles both .pth and .safetensors formats via the underlying Dia class.
    """
    global dia_model, model_config_instance, model_device, MODEL_LOADED, EXPECTED_SAMPLE_RATE

    if MODEL_LOADED:
        logger.info("Dia model already loaded.")
        return True

    # Get configuration values
    repo_id = get_model_repo_id()
    config_filename = get_model_config_filename()
    weights_filename = get_model_weights_filename()
    cache_path = get_model_cache_path()
    model_device = get_device()
    compute_dtype_str = get_compute_dtype(model_device)  # Determine compute dtype

    logger.info(f"Attempting to load Dia model:")
    logger.info(f"  Repo ID: {repo_id}")
    logger.info(f"  Config File: {config_filename}")
    logger.info(f"  Weights File: {weights_filename}")
    logger.info(f"  Cache Directory: {cache_path}")
    logger.info(f"  Target Device: {model_device}")
    logger.info(f"  Compute Dtype: {compute_dtype_str}")

    # Ensure cache directory exists
    try:
        os.makedirs(cache_path, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create cache directory '{cache_path}': {e}", exc_info=True
        )
        # Allow hf_hub_download to handle potential issues later
        pass

    try:
        start_time = time.time()

        # --- Download Model Files ---
        logger.info(
            f"Downloading/finding configuration file '{config_filename}' from repo '{repo_id}'..."
        )
        local_config_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            cache_dir=cache_path,
            # force_download=True # Uncomment to force redownload for testing
        )
        logger.info(f"Configuration file path: {local_config_path}")

        logger.info(
            f"Downloading/finding weights file '{weights_filename}' from repo '{repo_id}'..."
        )
        local_weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            cache_dir=cache_path,
            # force_download=True # Uncomment to force redownload for testing
        )
        logger.info(f"Weights file path: {local_weights_path}")

        # --- Load Model using the NEW Dia class method ---

        config = DiaConfig.load(local_config_path)
        if config is None:
            raise FileNotFoundError(
                f"Config file failed to load from {local_config_path}"
            )

        # Instantiate the Dia class
        dia_instance = Dia(config, compute_dtype=compute_dtype_str, device=model_device)

        # Load weights manually based on file type
        # Load to CPU first to potentially reduce GPU VRAM spike during loading
        logger.info(f"Loading weights from: {local_weights_path} to CPU RAM first...")
        map_location = torch.device("cpu")  # Load to CPU
        if local_weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            logger.info(
                "Detected .safetensors format. Loading using safetensors library."
            )
            state_dict = load_file(local_weights_path, device=str(map_location))
        elif local_weights_path.endswith(".pth"):
            logger.info("Detected .pth format. Loading using torch.load.")
            state_dict = torch.load(local_weights_path, map_location=map_location)
        else:
            raise ValueError(f"Unsupported weights file format: {weights_filename}")

        logger.info("Applying loaded weights to the model structure...")
        dia_instance.model.load_state_dict(state_dict)
        logger.info("Weights applied successfully.")

        # Move model to target device and set eval mode
        logger.info(f"Moving model to target device: {model_device}")
        dia_instance.model.to(model_device)
        dia_instance.model.eval()

        # Load DAC model (now handled within Dia class constructor or methods)
        logger.info("Loading associated DAC model...")
        dia_instance._load_dac_model()  # Call the internal method

        dia_model = dia_instance  # Assign to global variable
        model_config_instance = dia_model.config  # Store config if needed
        EXPECTED_SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # Use the constant from dia library

        end_time = time.time()
        logger.info(
            f"Dia model loaded successfully in {end_time - start_time:.2f} seconds."
        )
        MODEL_LOADED = True
        return True

    except FileNotFoundError as e:
        logger.error(
            f"Model loading failed: Required file not found. {e}", exc_info=True
        )
        MODEL_LOADED = False
        return False
    except ImportError:
        logger.critical(
            "Failed to load model: Dia package or its core dependencies not found.",
            exc_info=True,
        )
        MODEL_LOADED = False
        return False
    except Exception as e:
        logger.error(
            f"Error loading Dia model from repo '{repo_id}': {e}", exc_info=True
        )
        dia_model = None
        model_config_instance = None
        MODEL_LOADED = False
        return False


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


# --- Speech Generation ---


# --- Simplified Speech Generation Function ---
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
    Generates speech using the loaded Dia model by calling model.generate() for each chunk.
    This version explicitly ignores internal model state between chunks, potentially
    affecting coherence for long inputs. It follows the author's sample calling pattern per chunk.

    Args:
        (Same as original generate_speech)

    Returns:
        Tuple of (numpy_audio_array, sample_rate), or None on failure.
    """
    global dia_model  # Use the preloaded model instance

    if not MODEL_LOADED or dia_model is None or not model_device:
        logger.error("Dia model is not loaded. Cannot generate speech.")
        return None

    # Clear CUDA cache before starting new generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    monitor = PerformanceMonitor()
    monitor.record("Request received in engine (simple generate)")

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
