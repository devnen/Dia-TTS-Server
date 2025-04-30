# server.py
# Main FastAPI server for Dia TTS

import sys
import logging
import time
import os
import io
import uuid
import shutil
import yaml  # Keep yaml import for potential future use, though config handles it now
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Literal, List, Dict, Any
import webbrowser
import threading
import time

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    Form,
    UploadFile,
    File,
    BackgroundTasks,
    Depends,  # Added Depends for potential future use
)
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    HTMLResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np

# Internal imports
from config import (
    config_manager,  # Use the singleton instance
    get_host,
    get_port,
    get_output_path,
    get_reference_audio_path,
    get_model_cache_path,
    get_predefined_voices_path,
    get_model_repo_id,
    get_model_config_filename,
    get_model_weights_filename,
    get_whisper_model_name,
    # Generation default getters (still useful for API defaults)
    get_gen_default_speed_factor,
    get_gen_default_cfg_scale,
    get_gen_default_temperature,
    get_gen_default_top_p,
    get_gen_default_cfg_filter_top_k,
    get_gen_default_seed,
    get_gen_default_split_text,
    get_gen_default_chunk_size,
    CONFIG_FILE_PATH,
    ENV_FILE_PATH,
)

# Import updated request models
from models import OpenAITTSRequest, CustomTTSRequest, ErrorResponse
import engine
from engine import (
    load_model as load_dia_model,
    generate_speech,
    EXPECTED_SAMPLE_RATE,
)
from utils import (
    encode_audio,
    save_audio_to_file,
    PerformanceMonitor,
    sanitize_filename,
    get_valid_reference_files,  # Import helper from utils
    get_predefined_voices,  # Import helper from utils
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Reduce verbosity of noisy libraries if needed
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
# Set tqdm logging level higher to avoid interfering with progress bar
logging.getLogger("tqdm").setLevel(logging.WARNING)
# Set whisper logging level higher
logging.getLogger("whisper").setLevel(logging.WARNING)
# Set parselmouth logging level higher
logging.getLogger("parselmouth").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Global Variables & Constants ---
# Presets are now loaded directly in the UI route if needed, or managed differently
# PRESETS_FILE = "ui/presets.yaml" # No longer needed here
# loaded_presets: List[Dict[str, Any]] = [] # No longer needed here
startup_complete_event = threading.Event()

# --- Helper Functions (Moved relevant ones to utils.py) ---
# get_valid_reference_files is now in utils.py
# get_predefined_voices is now in utils.py


# --- Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    model_loaded_successfully = False
    try:
        logger.info("Starting Dia TTS server initialization...")
        # Config is loaded automatically by config_manager instance creation
        logger.info(
            f"Configuration loaded from {CONFIG_FILE_PATH if os.path.exists(CONFIG_FILE_PATH) else 'defaults/env'}."
        )

        # Ensure base directories exist based on loaded config
        os.makedirs(get_output_path(), exist_ok=True)
        os.makedirs(get_reference_audio_path(), exist_ok=True)
        os.makedirs(get_model_cache_path(), exist_ok=True)
        os.makedirs(
            get_predefined_voices_path(), exist_ok=True
        )  # Ensure voices dir exists
        os.makedirs("ui", exist_ok=True)

        # Load the main TTS model during startup
        if not load_dia_model():
            error_msg = "CRITICAL: Failed to load Dia model on startup. Server cannot function correctly."
            logger.critical(error_msg)
            # Allow server to start but log critical error. Endpoints will fail later.
        else:
            logger.info("Dia model loaded successfully.")
            model_loaded_successfully = True

            # Create and start a delayed browser opening thread only if model loaded
            host = get_host()
            port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host, port), daemon=True
            )
            browser_thread.start()

        logger.info("Application startup sequence finished. Signaling readiness.")
        startup_complete_event.set()

        yield  # Application runs here

    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}", exc_info=True)
        startup_complete_event.set()  # Ensure event is set even on error
    finally:
        logger.info("Application shutdown initiated...")
        # Add any specific cleanup needed
        logger.info("Application shutdown complete.")


def _delayed_browser_open(host, port):
    """Opens browser after a short delay to ensure server is ready"""
    try:
        startup_complete_event.wait(timeout=300)
        if not startup_complete_event.is_set():
            logger.warning(
                "Startup did not complete within timeout. Browser will not be opened automatically."
            )
            return
        time.sleep(2)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open browser at {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Dia TTS Server",
    description="Text-to-Speech server using the Dia model, providing API and Web UI.",
    version="1.4.0",  # Incremented version for config.yaml changes
    lifespan=lifespan,
)

# Check/Create necessary folders on startup (redundant with lifespan but safe)
folders_to_check = [
    get_output_path(),
    get_reference_audio_path(),
    get_model_cache_path(),
    get_predefined_voices_path(),
    "ui",
]
for folder in folders_to_check:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create directory '{folder}': {e}")


# --- Static Files and Templates ---
try:
    app.mount("/outputs", StaticFiles(directory=get_output_path()), name="outputs")
except RuntimeError as e:
    logger.error(
        f"Failed to mount /outputs directory '{get_output_path()}': {e}. Output files may not be accessible via URL."
    )
try:
    app.mount("/ui", StaticFiles(directory="ui"), name="ui_static")
except RuntimeError as e:
    logger.error(f"Failed to mount /ui directory: {e}. Web UI assets may not load.")

templates = Jinja2Templates(directory="ui")


# --- Configuration Routes (New YAML-based) ---
@app.post(
    "/save_settings",
    tags=["Configuration"],
    summary="Save partial configuration updates",
)
async def save_settings(request: Request):
    """
    Saves partial configuration updates (e.g., UI state, generation defaults,
    server settings) to the config.yaml file. Merges the update with the
    current configuration. Server restart is required for some settings (server, model, paths).
    """
    logger.info("Request received for /save_settings")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            # Determine if restart is likely needed based on keys updated
            restart_needed = any(
                k in partial_update for k in ["server", "model", "paths"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " Server restart required to apply all changes."

            return JSONResponse(
                content={"message": message, "restart_needed": restart_needed}
            )
        else:
            logger.error("Failed to save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to save configuration file."
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during save: {str(e)}"
        )


@app.post(
    "/reset_settings",
    tags=["Configuration"],
    summary="Reset configuration to defaults",
)
async def reset_settings():
    """
    Resets the configuration in config.yaml back to the hardcoded defaults,
    potentially overridden by values found in the .env file (if it exists).
    """
    logger.warning("Received request to reset configuration via API.")
    try:
        if config_manager.reset_and_save():
            logger.info(
                "Configuration reset to defaults (with .env overrides) and saved."
            )
            return JSONResponse(
                content={
                    "message": "Configuration reset successfully. Reload page or restart server."
                }
            )
        else:
            logger.error("Failed to reset and save configuration.")
            raise HTTPException(
                status_code=500, detail="Failed to reset configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during reset: {str(e)}"
        )


# --- Helper Endpoints for UI ---
@app.get(
    "/get_reference_files",
    tags=["UI Helpers"],
    summary="Get list of reference audio files",
)
async def get_reference_files_endpoint():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request received for /get_reference_files")
    try:
        files = get_valid_reference_files()
        return JSONResponse(content=files)
    except Exception as e:
        logger.error(f"Error getting reference files: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference files."
        )


@app.get(
    "/get_predefined_voices",
    tags=["UI Helpers"],
    summary="Get list of predefined voices",
)
async def get_predefined_voices_endpoint():
    """
    Returns a list of predefined voices found in the voices directory.
    Format: [{"display_name": "Formatted Name", "filename": "original_file.wav"}, ...]
    """
    logger.debug("Request received for /get_predefined_voices")
    try:
        voices = get_predefined_voices()
        return JSONResponse(content=voices)
    except Exception as e:
        logger.error(f"Error getting predefined voices: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices."
        )


# --- API Endpoints (TTS Generation) ---


@app.post(
    "/v1/audio/speech",
    response_class=StreamingResponse,
    tags=["TTS Generation"],
    summary="Generate speech (OpenAI compatible)",
    responses={
        200: {"content": {"audio/opus": {}, "audio/wav": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def openai_tts_endpoint(request: OpenAITTSRequest):
    """
    Generates speech audio from text, compatible with the OpenAI TTS API structure.
    Maps the 'voice' parameter to Dia's voice modes ('S1', 'S2', 'dialogue', or filename for clone).
    Uses default generation parameters from config.yaml unless overridden by future API extensions.
    """
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting or encountered an error.",
        )

    monitor = PerformanceMonitor()
    monitor.record("Request received")
    logger.info(
        f"Received OpenAI request: voice='{request.voice}', speed={request.speed}, format='{request.response_format}', seed={request.seed}"
    )
    logger.debug(f"Input text (start): '{request.input[:100]}...'")

    voice_mode = "single_s1"  # Default
    clone_ref_file = None
    predefined_voice_file = None  # Added for potential future mapping
    ref_path = get_reference_audio_path()
    predefined_path = get_predefined_voices_path()

    # Map OpenAI 'voice' parameter
    voice_param = request.voice.strip()

    # Check predefined voices first (case-insensitive match on display name or filename)
    # This requires getting the list of predefined voices
    predefined_voices = (
        get_predefined_voices()
    )  # List of {"display_name": "...", "filename": "..."}
    found_predefined = False
    for voice_info in predefined_voices:
        # Match against display name (case-insensitive) or filename (case-insensitive)
        if (
            voice_param.lower() == voice_info["display_name"].lower()
            or voice_param.lower() == voice_info["filename"].lower()
        ):
            voice_mode = "predefined"  # Treat as predefined internally
            predefined_voice_file = voice_info["filename"]  # Use the original filename
            logger.info(
                f"OpenAI request mapped to predefined voice: {predefined_voice_file}"
            )
            found_predefined = True
            break

    if not found_predefined:
        # If not predefined, check standard modes or reference files
        if voice_param.lower() == "dialogue":
            voice_mode = "dialogue"
        elif voice_param.lower() == "s1":
            voice_mode = "single_s1"
        elif voice_param.lower() == "s2":
            voice_mode = "single_s2"
        elif voice_param.lower().endswith((".wav", ".mp3")):
            # Check if file exists in reference audio path
            potential_path = os.path.join(ref_path, voice_param)
            if os.path.isfile(potential_path):
                voice_mode = "clone"
                clone_ref_file = voice_param
                logger.info(
                    f"OpenAI request mapped to clone mode with file: {clone_ref_file}"
                )
            else:
                logger.error(
                    f"Reference file '{voice_param}' specified in OpenAI request not found in '{ref_path}'. Cannot use clone mode."
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Reference audio file '{voice_param}' not found on server for cloning.",
                )
        else:
            logger.warning(
                f"Unrecognized OpenAI voice parameter '{voice_param}'. Defaulting voice mode to 'single_s1'."
            )
            voice_mode = "single_s1"  # Explicitly set default

    monitor.record("Parameters processed")

    try:
        # Determine the actual reference file path based on mode
        reference_file_for_engine = None
        if voice_mode == "clone" and clone_ref_file:
            reference_file_for_engine = os.path.join(ref_path, clone_ref_file)
        elif voice_mode == "predefined" and predefined_voice_file:
            reference_file_for_engine = os.path.join(
                predefined_path, predefined_voice_file
            )
            # For engine, treat predefined as a clone using the specific file
            voice_mode_for_engine = "clone"
        else:
            voice_mode_for_engine = voice_mode  # Use dialogue, single_s1, single_s2

        # Call the core engine function using mapped parameters and defaults from config.yaml
        result = generate_speech(
            text_to_process=request.input,
            voice_mode=voice_mode_for_engine,  # Pass mode adjusted for engine
            clone_reference_filename=reference_file_for_engine,  # Pass full path or None
            transcript=None,  # OpenAI API doesn't support explicit transcript
            speed_factor=request.speed,
            seed=request.seed,
            # Use Dia's configured defaults for other generation params
            max_tokens=None,
            cfg_scale=get_gen_default_cfg_scale(),
            temperature=get_gen_default_temperature(),
            top_p=get_gen_default_top_p(),
            cfg_filter_top_k=get_gen_default_cfg_filter_top_k(),
            split_text=get_gen_default_split_text(),  # Use configured default
            chunk_size=get_gen_default_chunk_size(),  # Use configured default
            # Use default post-processing settings
            enable_silence_trimming=True,
            enable_internal_silence_fix=True,
            enable_unvoiced_removal=True,
        )
        monitor.record("Generation complete")

        if result is None:
            logger.error("Speech generation failed (engine returned None).")
            if voice_mode_for_engine == "clone":
                raise HTTPException(
                    status_code=400,
                    detail="Cloning preparation failed. Check server logs (e.g., missing transcript/Whisper failure).",
                )
            else:
                raise HTTPException(status_code=500, detail="Speech generation failed.")

        audio_array, sample_rate = result

        if sample_rate != EXPECTED_SAMPLE_RATE:
            logger.warning(
                f"Engine returned sample rate {sample_rate}, expected {EXPECTED_SAMPLE_RATE}. Encoding will use {EXPECTED_SAMPLE_RATE}."
            )
            sample_rate = EXPECTED_SAMPLE_RATE

        encoded_audio = encode_audio(audio_array, sample_rate, request.response_format)
        monitor.record("Audio encoding complete")

        if encoded_audio is None:
            logger.error(f"Failed to encode audio to format: {request.response_format}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {request.response_format}",
            )

        media_type = "audio/opus" if request.response_format == "opus" else "audio/wav"
        logger.info(
            f"Successfully generated {len(encoded_audio)} bytes in format {request.response_format}"
        )
        logger.debug(monitor.report())

        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException as http_exc:
        logger.error(f"HTTP exception during OpenAI request: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing OpenAI TTS request: {e}", exc_info=True)
        logger.debug(monitor.report())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post(
    "/tts",
    response_class=StreamingResponse,
    tags=["TTS Generation"],
    summary="Generate speech (Custom parameters)",
    responses={
        200: {"content": {"audio/opus": {}, "audio/wav": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def custom_tts_endpoint(request: CustomTTSRequest):
    """
    Generates speech audio from text using explicit Dia parameters,
    including text splitting options and optional transcript for cloning.
    Handles 'predefined' voice mode by mapping to 'clone' with correct path.
    """
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting or encountered an error.",
        )

    monitor = PerformanceMonitor()
    monitor.record("Request received")
    logger.info(
        f"Received custom TTS request: mode='{request.voice_mode}', format='{request.output_format}', seed={request.seed}, split={request.split_text}, chunk={request.chunk_size}, transcript_provided={request.transcript is not None}"
    )
    logger.debug(f"Input text (start): '{request.text[:100]}...'")
    logger.debug(
        f"Params: max_tokens={request.max_tokens}, cfg={request.cfg_scale}, temp={request.temperature}, top_p={request.top_p}, speed={request.speed_factor}, top_k={request.cfg_filter_top_k}"
    )

    reference_file_for_engine = None
    voice_mode_for_engine = request.voice_mode  # Start with requested mode

    if request.voice_mode == "clone":
        if not request.clone_reference_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'clone_reference_filename' for clone mode.",
            )
        ref_path = get_reference_audio_path()
        potential_path = os.path.join(ref_path, request.clone_reference_filename)
        if not os.path.isfile(potential_path):
            logger.error(
                f"Reference audio file not found for clone mode: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file not found: {request.clone_reference_filename}",
            )
        reference_file_for_engine = potential_path  # Use full path
        logger.info(
            f"Custom request using clone mode with file: {request.clone_reference_filename}"
        )

    elif request.voice_mode == "predefined":
        if (
            not request.clone_reference_filename
        ):  # Use the same field to pass the predefined filename
            raise HTTPException(
                status_code=400,
                detail="Missing 'clone_reference_filename' (expected predefined voice filename) for predefined mode.",
            )
        predefined_path = get_predefined_voices_path()
        potential_path = os.path.join(predefined_path, request.clone_reference_filename)
        if not os.path.isfile(potential_path):
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file not found: {request.clone_reference_filename}",
            )
        reference_file_for_engine = potential_path  # Use full path
        voice_mode_for_engine = "clone"  # Treat as clone for engine
        logger.info(
            f"Custom request using predefined mode, mapped to clone with file: {request.clone_reference_filename}"
        )

    # For dialogue, single_s1, single_s2, reference_file_for_engine remains None

    monitor.record("Parameters processed")

    try:
        # Call the core engine function with all parameters from the request
        result = generate_speech(
            text_to_process=request.text,
            voice_mode=voice_mode_for_engine,  # Use potentially adjusted mode
            clone_reference_filename=reference_file_for_engine,  # Pass full path or None
            transcript=request.transcript,  # Pass the optional transcript
            max_tokens=request.max_tokens,
            cfg_scale=request.cfg_scale,
            temperature=request.temperature,
            top_p=request.top_p,
            speed_factor=request.speed_factor,
            cfg_filter_top_k=request.cfg_filter_top_k,
            seed=request.seed,
            split_text=request.split_text,
            chunk_size=request.chunk_size,
            # Use default post-processing settings
            enable_silence_trimming=True,
            enable_internal_silence_fix=True,
            enable_unvoiced_removal=True,
        )
        monitor.record("Generation complete")

        if result is None:
            logger.error("Speech generation failed (engine returned None).")
            if voice_mode_for_engine == "clone":
                raise HTTPException(
                    status_code=400,
                    detail="Cloning preparation failed. Check server logs (e.g., missing transcript/Whisper failure).",
                )
            else:
                raise HTTPException(status_code=500, detail="Speech generation failed.")

        audio_array, sample_rate = result

        if sample_rate != EXPECTED_SAMPLE_RATE:
            logger.warning(
                f"Engine returned sample rate {sample_rate}, expected {EXPECTED_SAMPLE_RATE}. Encoding will use {EXPECTED_SAMPLE_RATE}."
            )
            sample_rate = EXPECTED_SAMPLE_RATE

        encoded_audio = encode_audio(audio_array, sample_rate, request.output_format)
        monitor.record("Audio encoding complete")

        if encoded_audio is None:
            logger.error(f"Failed to encode audio to format: {request.output_format}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {request.output_format}",
            )

        media_type = "audio/opus" if request.output_format == "opus" else "audio/wav"
        logger.info(
            f"Successfully generated {len(encoded_audio)} bytes in format {request.output_format}"
        )
        logger.debug(monitor.report())

        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException as http_exc:
        logger.error(f"HTTP exception during custom TTS request: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing custom TTS request: {e}", exc_info=True)
        logger.debug(monitor.report())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Web UI Endpoints ---


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main TTS web interface."""
    logger.info("Serving TTS Web UI (index.html)")
    try:
        # Load current state from config manager
        full_config = config_manager.get_all()
        reference_files = get_valid_reference_files()
        predefined_voices = get_predefined_voices()

        predefined_voices_error = None
        if not predefined_voices:
            predefined_voices_error = (
                "No predefined voices found in './voices' directory."
            )
            logger.warning(predefined_voices_error)
            # Ensure UI state reflects this if default was 'predefined'
            if full_config.get("ui_state", {}).get("last_voice_mode") == "predefined":
                full_config.setdefault("ui_state", {})[
                    "last_voice_mode"
                ] = "dialogue"  # Fallback mode
                full_config.setdefault("ui_state", {})["last_predefined_voice"] = None

        # Load presets from file
        loaded_presets = []  # Default to empty list
        presets_file_path = "ui/presets.yaml"
        try:
            if os.path.exists(presets_file_path):
                with open(presets_file_path, "r", encoding="utf-8") as f:
                    yaml_content = yaml.safe_load(f)
                    if isinstance(yaml_content, list):
                        loaded_presets = yaml_content
                        logger.info(
                            f"Successfully loaded {len(loaded_presets)} presets from {presets_file_path}."
                        )
                    else:
                        logger.error(
                            f"Invalid format in {presets_file_path}. Expected a list, got {type(yaml_content)}. No presets loaded."
                        )
            else:
                logger.warning(
                    f"Presets file not found: {presets_file_path}. No presets will be available."
                )
        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing presets YAML file '{presets_file_path}': {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Error loading presets file '{presets_file_path}': {e}", exc_info=True
            )
        # Ensure loaded_presets is always a list before passing to template
        if not isinstance(loaded_presets, list):
            loaded_presets = []

        # Create a single object for all generation result data
        initial_gen_result = {
            "outputUrl": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        # Pass the necessary data to the template
        template_context = {
            "request": request,
            "config": full_config,  # Pass the whole config object
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "predefined_voices_error": predefined_voices_error,
            "presets": loaded_presets,
            # Pass the single generation result object
            "initial_gen_result": initial_gen_result,
            # Keep error/success for notifications
            "error": None,
            "success": None,
        }

        return templates.TemplateResponse("index.html", template_context)

    except Exception as e:
        logger.error(f"Error rendering Web UI: {e}", exc_info=True)
        # Fallback error page or simple response
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. Check server logs.</p></body></html>",
            status_code=500,
        )


@app.post("/web/generate", response_class=HTMLResponse, include_in_schema=False)
async def handle_web_ui_generate(
    request: Request,
    text: str = Form(...),
    voice_mode: Literal["predefined", "clone", "dialogue"] = Form(
        ...
    ),  # Add predefined mode support
    predefined_voice_select: Optional[str] = Form(
        None
    ),  # Filename for predefined voice
    clone_reference_select: Optional[str] = Form(None),  # Filename for clone reference
    # Generation parameters from form
    speed_factor: float = Form(...),
    cfg_scale: float = Form(...),
    temperature: float = Form(...),
    top_p: float = Form(...),
    cfg_filter_top_k: int = Form(...),
    seed: int = Form(...),
    split_text: Optional[bool] = Form(False),
    chunk_size: Optional[int] = Form(120),  # Match default from config/UI
):
    """
    Handles the generation request from the web UI form.
    Validates inputs, calls the engine, saves the output, and re-renders
    the UI template with results or errors.
    """
    # --- Initial check: Is the TTS engine ready? ---
    if not engine.MODEL_LOADED:
        logger.error("Web UI generation request received, but model is not loaded.")
        # Re-render form with error message if model is not ready.
        # Fetch necessary data again to populate the template correctly.
        error_context = {
            "request": request,
            "error": "Model is not loaded. Please wait or check server logs.",
            "config": config_manager.get_all(),
            "reference_files": get_valid_reference_files(),
            "predefined_voices": get_predefined_voices(),
            "predefined_voices_error": (
                None if get_predefined_voices() else "No predefined voices found."
            ),
            "presets": [],  # Simplified error response
            "submitted_text": text,
            "submitted_voice_mode": voice_mode,
            "submitted_predefined_voice": predefined_voice_select,
            "submitted_clone_file": clone_reference_select,
            "submitted_gen_params": {
                "speed_factor": speed_factor,
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "top_p": top_p,
                "cfg_filter_top_k": cfg_filter_top_k,
                "seed": seed,
                "split_text": split_text,
                "chunk_size": chunk_size,
            },
            "success": None,
            # Provide a default initial_gen_result even for this error case
            "initial_gen_result": {
                "outputUrl": None,
                "genTime": None,
                "submittedVoiceMode": voice_mode,
                "submittedPredefinedVoice": predefined_voice_select,
                "submittedCloneFile": clone_reference_select,
            },
            "output_file_url": None,
            "generation_time": None,
        }
        return templates.TemplateResponse("index.html", error_context, status_code=503)

    # --- Start processing the valid request ---
    logger.info(
        f"Web UI generation request: mode='{voice_mode}', seed={seed}, split={split_text}, chunk={chunk_size}"
    )
    monitor = PerformanceMonitor()
    monitor.record("Web request received")

    # Initialize variables
    output_file_url = None
    generation_time = None
    error_message = None
    success_message = None
    reference_file_for_engine = None
    voice_mode_for_engine = voice_mode
    effective_filename_for_log = None

    # --- Pre-generation Validation & Path Setup ---
    if not text.strip():
        error_message = "Please enter some text to synthesize."

    if not error_message and voice_mode == "clone":
        if not clone_reference_select or clone_reference_select == "none":
            error_message = "Please select a reference audio file for clone mode."
        else:
            ref_path = get_reference_audio_path()
            potential_path = os.path.join(ref_path, clone_reference_select)
            if not os.path.isfile(potential_path):
                error_message = f"Selected reference file '{clone_reference_select}' no longer exists. Please refresh or upload."
                clone_reference_select = None
            else:
                reference_file_for_engine = potential_path
                effective_filename_for_log = clone_reference_select
                logger.info(f"Using selected reference file: {clone_reference_select}")

    elif not error_message and voice_mode == "predefined":
        if not predefined_voice_select or predefined_voice_select == "none":
            error_message = "Please select a predefined voice."
        else:
            predefined_path = get_predefined_voices_path()
            potential_path = os.path.join(predefined_path, predefined_voice_select)
            if not os.path.isfile(potential_path):
                error_message = f"Selected predefined voice file '{predefined_voice_select}' no longer exists."
                predefined_voice_select = None
            else:
                reference_file_for_engine = potential_path
                voice_mode_for_engine = "clone"
                effective_filename_for_log = predefined_voice_select
                logger.info(
                    f"Using selected predefined voice file: {predefined_voice_select}"
                )

    elif not error_message and voice_mode == "dialogue":
        if not ("[S1]" in text or "[S2]" in text):
            logger.info(
                "Dialogue mode selected, but no [S1]/[S2] tags found. Treating as single_s1."
            )
            voice_mode_for_engine = "single_s1"

    # --- Save UI State (if no validation errors yet) ---
    if not error_message:
        try:
            ui_state_update = {
                "ui_state": {
                    "last_text": text,
                    "last_voice_mode": voice_mode,
                    "last_predefined_voice": (
                        predefined_voice_select if voice_mode == "predefined" else None
                    ),
                    "last_reference_file": (
                        clone_reference_select if voice_mode == "clone" else None
                    ),
                    "last_seed": seed,
                    "last_chunk_size": chunk_size,
                    "last_split_text_enabled": split_text,
                }
            }
            if not config_manager.update_and_save(ui_state_update):
                logger.warning("Failed to save UI state update before generation.")
            else:
                logger.debug("Saved UI state update before generation.")
        except Exception as save_err:
            logger.warning(f"Error saving UI state before generation: {save_err}")

    # --- Handle Validation Errors: Re-render page ---
    if error_message:
        logger.warning(f"Web UI validation error: {error_message}")
        # Fetch latest data for template rendering
        full_config = config_manager.get_all()
        reference_files = get_valid_reference_files()
        predefined_voices = get_predefined_voices()
        loaded_presets = []
        try:  # Reload presets
            presets_file_path = "ui/presets.yaml"
            if os.path.exists(presets_file_path):
                with open(presets_file_path, "r", encoding="utf-8") as f:
                    loaded_presets = yaml.safe_load(f) or []
                    if not isinstance(loaded_presets, list):
                        loaded_presets = []
        except Exception as preset_load_err:
            logger.warning(
                f"Could not reload presets for error page: {preset_load_err}"
            )
            loaded_presets = []

        # Prepare context for the error response template
        error_context = {
            "request": request,
            "error": error_message,
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "predefined_voices_error": (
                None if predefined_voices else "No predefined voices found."
            ),
            "presets": loaded_presets,
            "submitted_text": text,
            "submitted_voice_mode": voice_mode,
            "submitted_predefined_voice": predefined_voice_select,
            "submitted_clone_file": clone_reference_select,
            "submitted_gen_params": {
                "speed_factor": speed_factor,
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "top_p": top_p,
                "cfg_filter_top_k": cfg_filter_top_k,
                "seed": seed,
                "split_text": split_text,
                "chunk_size": chunk_size,
            },
            "success": None,
            # Also provide default initial_gen_result here
            "initial_gen_result": {
                "outputUrl": None,
                "genTime": None,
                "submittedVoiceMode": voice_mode,
                "submittedPredefinedVoice": predefined_voice_select,
                "submittedCloneFile": clone_reference_select,
            },
            "output_file_url": None,
            "generation_time": None,
        }
        return templates.TemplateResponse(
            "index.html", error_context, status_code=400
        )  # Bad Request

    # --- Generation ---
    try:
        monitor.record("Parameters processed, starting generation")
        actual_chunk_size = chunk_size if chunk_size is not None else 120

        # Adjust split_text based on length
        if split_text and len(text) < actual_chunk_size * 2:
            logger.info(
                f"Backend disabling split_text as text length ({len(text)}) is < 2x chunk size ({actual_chunk_size})."
            )
            split_text = False

        # Call the core speech generation function
        result = generate_speech(
            text_to_process=text,
            voice_mode=voice_mode_for_engine,
            clone_reference_filename=reference_file_for_engine,
            transcript=None,
            speed_factor=speed_factor,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            seed=seed,
            split_text=split_text,
            chunk_size=actual_chunk_size,
            max_tokens=None,
            enable_silence_trimming=True,
            enable_internal_silence_fix=True,
            enable_unvoiced_removal=True,
        )
        monitor.record("Generation complete")

        if result:
            # Process successful generation
            audio_array, sample_rate = result
            output_path_base = get_output_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_tag = voice_mode_for_engine

            # Create descriptive filename tag
            if voice_mode_for_engine == "clone" and effective_filename_for_log:
                safe_ref_name = sanitize_filename(
                    os.path.splitext(effective_filename_for_log)[0]
                )
                prefix = "clone" if voice_mode == "clone" else "voice"
                mode_tag = f"{prefix}_{safe_ref_name[:20]}"
            elif voice_mode_for_engine == "dialogue":
                mode_tag = "dialogue"
            elif voice_mode_for_engine == "single_s1":
                mode_tag = "single_s1"
            elif voice_mode_for_engine == "single_s2":
                mode_tag = "single_s2"
            else:
                mode_tag = "output"

            output_filename = f"{mode_tag}_{timestamp}.wav"
            output_filepath = os.path.join(output_path_base, output_filename)

            # Save the audio file
            saved = save_audio_to_file(audio_array, sample_rate, output_filepath)
            monitor.record("Audio saved")

            if saved:
                output_file_url = f"/outputs/{output_filename}"
                generation_time = monitor.events[-1][1] - monitor.start_time
                success_message = "Audio generated successfully!"
                logger.info(f"Web UI generated audio saved to: {output_filepath}")
            else:
                error_message = "Failed to save generated audio file. Check server permissions and path."
                logger.error(
                    f"Failed to save audio file from web UI request to {output_filepath}."
                )
        else:
            # Handle generation failure
            if voice_mode_for_engine == "clone":
                error_message = "Cloning preparation failed. Check server logs (e.g., missing transcript .txt, Whisper failure, audio issues)."
            else:
                error_message = "Speech generation failed (engine returned None). Check server logs."
            logger.error(
                f"Speech generation failed. Engine Mode: {voice_mode_for_engine}, UI Mode: {voice_mode}"
            )

    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Error processing web UI TTS request: {e}", exc_info=True)
        error_message = f"An unexpected error occurred during generation: {str(e)}"

    # Log performance
    logger.debug(monitor.report())

    # --- Prepare Data for Template Rendering ---

    # Construct the result dictionary for JavaScript (MUST be defined)
    initial_gen_result = {
        "outputUrl": output_file_url,
        "genTime": f"{generation_time:.2f}" if generation_time is not None else None,
        "submittedVoiceMode": voice_mode,
        "submittedPredefinedVoice": (
            predefined_voice_select if voice_mode == "predefined" else None
        ),
        "submittedCloneFile": clone_reference_select if voice_mode == "clone" else None,
    }

    # Fetch latest state data for rendering
    full_config = config_manager.get_all()
    reference_files = get_valid_reference_files()
    predefined_voices = get_predefined_voices()
    loaded_presets = []
    try:  # Reload presets
        presets_file_path = "ui/presets.yaml"
        if os.path.exists(presets_file_path):
            with open(presets_file_path, "r", encoding="utf-8") as f:
                loaded_presets = yaml.safe_load(f) or []
                if not isinstance(loaded_presets, list):
                    loaded_presets = []
    except Exception as preset_load_err:
        logger.warning(f"Could not reload presets for results page: {preset_load_err}")
        loaded_presets = []

    # Determine HTTP status code
    status_code = 200 if success_message else 500
    if error_message and "Cloning preparation failed" in error_message:
        status_code = 400

    # Prepare submitted generation parameters for form repopulation
    submitted_gen_params = {
        "speed_factor": speed_factor,
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "top_p": top_p,
        "cfg_filter_top_k": cfg_filter_top_k,
        "seed": seed,
        "split_text": split_text,  # Use potentially adjusted value
        "chunk_size": chunk_size if chunk_size is not None else 120,
    }

    # Assemble the complete context for Jinja2
    template_context = {
        "request": request,
        "error": error_message,
        "success": success_message,
        "initial_gen_result": initial_gen_result,  # <-- Pass the structured result data
        "config": full_config,
        "reference_files": reference_files,
        "predefined_voices": predefined_voices,
        "predefined_voices_error": (
            None if predefined_voices else "No predefined voices found."
        ),
        "presets": loaded_presets,
        # Submitted values for form repopulation
        "submitted_text": text,
        "submitted_voice_mode": voice_mode,
        "submitted_predefined_voice": predefined_voice_select,
        "submitted_clone_file": clone_reference_select,
        "submitted_gen_params": submitted_gen_params,
        # Individual result vars (optional, as JS uses initial_gen_result)
        "output_file_url": output_file_url,
        "generation_time": f"{generation_time:.2f}" if generation_time else None,
    }

    # Render and return the HTML response
    return templates.TemplateResponse(
        "index.html",
        template_context,  # Pass the complete context dictionary
        status_code=status_code,  # Set appropriate HTTP status
    )


# --- Reference Audio Upload Endpoint ---
@app.post(
    "/upload_reference", tags=["UI Helpers"], summary="Upload reference audio files"
)
async def upload_reference_audio(files: List[UploadFile] = File(...)):
    """Handles uploading of reference audio files (.wav, .mp3) for voice cloning."""
    logger.info(f"Received request to upload {len(files)} reference audio file(s).")
    ref_path = get_reference_audio_path()
    uploaded_filenames = []
    errors = []
    allowed_mime_types = [
        "audio/wav",
        "audio/mpeg",
        "audio/x-wav",
        "audio/mp3",
        "audio/wave",
        "audio/x-pn-wav",
    ]
    allowed_extensions = [".wav", ".mp3"]

    for file in files:
        try:
            if not file.filename:
                errors.append("Received file with no filename.")
                continue

            safe_filename = sanitize_filename(file.filename)
            _, ext = os.path.splitext(safe_filename)
            if ext.lower() not in allowed_extensions:
                errors.append(
                    f"File '{file.filename}' has unsupported extension '{ext}'. Allowed: {allowed_extensions}"
                )
                continue

            if file.content_type not in allowed_mime_types:
                logger.warning(
                    f"File '{file.filename}' has unexpected content type '{file.content_type}'. Allowed: {allowed_mime_types}. Proceeding based on extension."
                )

            destination_path = os.path.join(ref_path, safe_filename)

            if os.path.exists(destination_path):
                logger.warning(
                    f"Reference file '{safe_filename}' already exists. Skipping upload."
                )
                if safe_filename not in uploaded_filenames:
                    uploaded_filenames.append(safe_filename)  # Report as available
                continue

            try:
                file_content = await file.read()
                with open(destination_path, "wb") as buffer:
                    buffer.write(file_content)
                logger.info(f"Successfully saved reference file: {destination_path}")
                uploaded_filenames.append(safe_filename)
            except Exception as save_exc:
                errors.append(f"Failed to save file '{safe_filename}': {save_exc}")
                logger.error(
                    f"Failed to save uploaded file '{safe_filename}' to '{destination_path}': {save_exc}",
                    exc_info=True,
                )
            finally:
                await file.close()

        except Exception as e:
            errors.append(
                f"Error processing file '{getattr(file, 'filename', 'unknown')}': {e}"
            )
            logger.error(
                f"Unexpected error processing uploaded file: {e}", exc_info=True
            )
            if file:
                await file.close()

    updated_file_list = get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames,
        "all_reference_files": updated_file_list,
        "errors": errors,
    }
    status_code = 200 if not errors or len(errors) < len(files) else 400
    if errors:
        logger.warning(f"Upload completed with errors: {errors}")

    return JSONResponse(content=response_data, status_code=status_code)


# --- Health Check Endpoint ---
@app.get("/health", tags=["Server Status"], summary="Check server health")
async def health_check():
    """Basic health check, indicates if the server is running and if the model is loaded."""
    startup_complete_event.wait(timeout=0.5)
    current_model_status = getattr(engine, "MODEL_LOADED", False)
    logger.debug(f"Health check returning model_loaded status: {current_model_status}")
    return {"status": "healthy", "model_loaded": current_model_status}


# --- Main Execution ---
if __name__ == "__main__":
    host = get_host()
    port = get_port()
    logger.info(f"Starting Dia TTS server on {host}:{port}")
    logger.info(
        f"Configuration will be read from/written to: {os.path.abspath(CONFIG_FILE_PATH)}"
    )
    if not os.path.exists(CONFIG_FILE_PATH) and ENV_FILE_PATH:
        logger.info(
            f"'{CONFIG_FILE_PATH}' not found. Will attempt initial seeding from: {ENV_FILE_PATH}"
        )
    elif not os.path.exists(CONFIG_FILE_PATH):
        logger.info(f"'{CONFIG_FILE_PATH}' not found. Will create using defaults.")

    # Log key settings read from config
    logger.info(f"Model Repository: {get_model_repo_id()}")
    logger.info(f"Model Weights File: {get_model_weights_filename()}")
    logger.info(f"Model Cache Path: {get_model_cache_path()}")
    logger.info(f"Reference Audio Path: {get_reference_audio_path()}")
    logger.info(f"Predefined Voices Path: {get_predefined_voices_path()}")
    logger.info(f"Output Path: {get_output_path()}")
    logger.info(f"Whisper Model: {get_whisper_model_name()}")

    display_host = "localhost" if host == "0.0.0.0" else host
    logger.info(f"Web UI will be available at http://{display_host}:{port}/")
    logger.info(f"API Docs available at http://{display_host}:{port}/docs")

    # Ensure UI directory and index.html exist for UI
    ui_dir = "ui"
    index_file = os.path.join(ui_dir, "index.html")
    if not os.path.isdir(ui_dir) or not os.path.isfile(index_file):
        logger.warning(
            f"'{ui_dir}' directory or '{index_file}' not found. Web UI may not work."
        )
        os.makedirs(ui_dir, exist_ok=True)
        if not os.path.isfile(index_file):
            try:
                with open(index_file, "w") as f:
                    f.write("<html><body>Web UI template missing.</body></html>")
                logger.info(f"Created dummy {index_file}.")
            except Exception as e:
                logger.error(f"Failed to create dummy {index_file}: {e}")

    # Run Uvicorn server
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,  # Keep reload=False for production/stability
        lifespan="on",
        log_level="info",
        workers=1,  # Essential for stable in-memory config and model state
    )
