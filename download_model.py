# download_model.py
# Utility script to download the Dia model and dependencies without starting the server.

import logging
import os
import engine  # Import the engine module to trigger its loading logic

# Import Whisper for model download check
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper library not found. Cannot download Whisper model.")

# Configure basic logging for the script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ModelDownloader")

if __name__ == "__main__":
    logger.info("--- Starting Dia & Whisper Model Download ---")

    # Ensure cache directory exists (redundant if engine.load_model does it, but safe)
    try:
        from config import get_model_cache_path, get_whisper_model_name

        cache_path = get_model_cache_path()
        os.makedirs(cache_path, exist_ok=True)
        logger.info(
            f"Ensured model cache directory exists: {os.path.abspath(cache_path)}"
        )
    except Exception as e:
        logger.warning(f"Could not ensure cache directory exists: {e}")
        cache_path = None  # Ensure cache_path is defined or None

    # Trigger the Dia model loading function from the engine
    logger.info("Calling engine.load_model() to initiate Dia download if necessary...")
    dia_success = engine.load_model()

    if dia_success:
        logger.info("--- Dia model download/load process completed successfully ---")
    else:
        logger.error(
            "--- Dia model download/load process failed. Check logs for details. ---"
        )
        # Optionally exit if Dia fails, or continue to try Whisper
        # exit(1)

    # --- Download Whisper Model ---
    whisper_success = False
    if WHISPER_AVAILABLE and cache_path:
        whisper_model_name = get_whisper_model_name()
        logger.info(f"Attempting to download Whisper model '{whisper_model_name}'...")
        try:
            # Use download_root to specify our cache directory
            whisper.load_model(whisper_model_name, download_root=cache_path)
            logger.info(
                f"Whisper model '{whisper_model_name}' downloaded/found successfully in {cache_path}."
            )
            whisper_success = True
        except Exception as e:
            logger.error(
                f"Failed to download/load Whisper model '{whisper_model_name}': {e}",
                exc_info=True,
            )
    elif not WHISPER_AVAILABLE:
        logger.warning(
            "Skipping Whisper model download: Whisper library not installed."
        )
    elif not cache_path:
        logger.warning(
            "Skipping Whisper model download: Cache path could not be determined."
        )

    # --- Final Status ---
    if dia_success and whisper_success:
        logger.info("--- All required models downloaded/verified successfully ---")
    elif dia_success:
        logger.warning(
            "--- Dia model OK, but Whisper model download failed or was skipped ---"
        )
    elif whisper_success:
        logger.warning("--- Whisper model OK, but Dia model download failed ---")
    else:
        logger.error(
            "--- Both Dia and Whisper model downloads failed or were skipped ---"
        )
        exit(1)  # Exit with error code if essential models failed

    logger.info("You can now start the server using 'python server.py'")
