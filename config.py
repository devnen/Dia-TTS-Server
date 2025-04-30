# config.py
# Configuration management for Dia TTS server using YAML file.

import os
import logging
import yaml
import shutil
from copy import deepcopy
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple

# Use dotenv for initial seeding ONLY if config.yaml is missing
from dotenv import load_dotenv, find_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
CONFIG_FILE_PATH = "config.yaml"
ENV_FILE_PATH = find_dotenv()  # Find .env file location

# --- Default Configuration Structure ---
# This defines the expected structure and default values for config.yaml
# It's crucial to keep this up-to-date with all expected keys.
DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8003,
    },
    "model": {
        "repo_id": "ttj/dia-1.6b-safetensors",
        "config_filename": "config.json",
        "weights_filename": "dia-v0_1_bf16.safetensors",
        "whisper_model_name": "small.en",  # Added Whisper model here
    },
    "paths": {
        "model_cache": "./model_cache",
        "reference_audio": "./reference_audio",
        "output": "./outputs",
        "voices": "./voices",  # Added predefined voices path
    },
    "generation_defaults": {
        "speed_factor": 1.0,  # Changed default to 1.0
        "cfg_scale": 3.0,
        "temperature": 1.3,
        "top_p": 0.95,
        "cfg_filter_top_k": 35,
        "seed": 42,
        "split_text": True,  # Default split enabled
        "chunk_size": 120,  # Default chunk size
    },
    "ui_state": {
        "last_text": "",
        "last_voice_mode": "predefined",  # Default to predefined
        "last_predefined_voice": None,  # Store original filename (e.g., "Michael_Emily.wav")
        "last_reference_file": None,
        "last_seed": 42,
        "last_chunk_size": 120,
        "last_split_text_enabled": True,
        "hide_chunk_warning": False,
        "hide_generation_warning": False,
    },
}

# Mapping from .env variable names to config.yaml nested keys
# Used ONLY during initial seeding if config.yaml is missing.
ENV_TO_YAML_MAP: Dict[str, Tuple[List[str], type]] = {
    # Server
    "HOST": (["server", "host"], str),
    "PORT": (["server", "port"], int),
    # Model
    "DIA_MODEL_REPO_ID": (["model", "repo_id"], str),
    "DIA_MODEL_CONFIG_FILENAME": (["model", "config_filename"], str),
    "DIA_MODEL_WEIGHTS_FILENAME": (["model", "weights_filename"], str),
    "WHISPER_MODEL_NAME": (["model", "whisper_model_name"], str),
    # Paths
    "DIA_MODEL_CACHE_PATH": (["paths", "model_cache"], str),
    "REFERENCE_AUDIO_PATH": (["paths", "reference_audio"], str),
    "OUTPUT_PATH": (["paths", "output"], str),
    # Generation Defaults
    "GEN_DEFAULT_SPEED_FACTOR": (["generation_defaults", "speed_factor"], float),
    "GEN_DEFAULT_CFG_SCALE": (["generation_defaults", "cfg_scale"], float),
    "GEN_DEFAULT_TEMPERATURE": (["generation_defaults", "temperature"], float),
    "GEN_DEFAULT_TOP_P": (["generation_defaults", "top_p"], float),
    "GEN_DEFAULT_CFG_FILTER_TOP_K": (["generation_defaults", "cfg_filter_top_k"], int),
    "GEN_DEFAULT_SEED": (["generation_defaults", "seed"], int),
    # Note: split_text and chunk_size defaults are not typically in .env
    # Note: ui_state is never loaded from .env
}


def _deep_merge_dicts(source: Dict, destination: Dict) -> Dict:
    """
    Recursively merges source dict into destination dict.
    Modifies destination in place.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            _deep_merge_dicts(value, node)
        else:
            destination[key] = value
    return destination


def _set_nested_value(d: Dict, keys: List[str], value: Any):
    """Sets a value in a nested dictionary using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested_value(d: Dict, keys: List[str], default: Any = None) -> Any:
    """Gets a value from a nested dictionary using a list of keys."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


class YamlConfigManager:
    """Manages configuration for the TTS server using a YAML file."""

    def __init__(self):
        """Initialize the configuration manager by loading the config."""
        self.config: Dict[str, Any] = {}
        self._lock = Lock()  # Lock for thread-safe file writing
        self.load_config()

    def _load_defaults(self) -> Dict[str, Any]:
        """Returns a deep copy of the hardcoded default configuration."""
        return deepcopy(DEFAULT_CONFIG)

    def _load_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads .env file (if found) and overrides values in the provided config_dict.
        Used ONLY during initial seeding or reset.
        """
        if not ENV_FILE_PATH:
            logger.info("No .env file found, skipping environment variable overrides.")
            return config_dict

        logger.info(f"Loading environment variables from: {ENV_FILE_PATH}")
        # Load .env variables into os.environ temporarily
        load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)

        env_values_applied = 0
        for env_var, (yaml_path, target_type) in ENV_TO_YAML_MAP.items():
            env_value_str = os.environ.get(env_var)
            if env_value_str is not None:
                try:
                    # Attempt type conversion
                    if target_type is bool:
                        converted_value = env_value_str.lower() in (
                            "true",
                            "1",
                            "t",
                            "yes",
                            "y",
                        )
                    else:
                        converted_value = target_type(env_value_str)

                    _set_nested_value(config_dict, yaml_path, converted_value)
                    logger.debug(
                        f"Applied .env override: {'->'.join(yaml_path)} = {converted_value}"
                    )
                    env_values_applied += 1
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not apply .env override for '{env_var}'. Invalid value '{env_value_str}' for type {target_type.__name__}. Using default. Error: {e}"
                    )
            # Clean up the loaded env var from os.environ if desired, though it's usually harmless
            # if env_var in os.environ:
            #     del os.environ[env_var]

        if env_values_applied > 0:
            logger.info(f"Applied {env_values_applied} overrides from .env file.")
        else:
            logger.info("No applicable overrides found in .env file.")

        return config_dict

    def load_config(self):
        """
        Loads configuration from config.yaml.
        If config.yaml doesn't exist, creates it by seeding from .env (if found)
        and hardcoded defaults.
        Ensures all default keys are present in the loaded config.
        """
        with self._lock:  # Ensure loading process is atomic if called concurrently (unlikely at startup)
            loaded_config = self._load_defaults()  # Start with defaults

            if os.path.exists(CONFIG_FILE_PATH):
                logger.info(f"Loading configuration from: {CONFIG_FILE_PATH}")
                try:
                    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f)
                        if isinstance(yaml_data, dict):
                            # Merge loaded data onto defaults to ensure all keys exist
                            # and to add new default keys if the file is old.
                            loaded_config = _deep_merge_dicts(yaml_data, loaded_config)
                            logger.info("Successfully loaded and merged config.yaml.")
                        else:
                            logger.error(
                                f"Invalid format in {CONFIG_FILE_PATH}. Expected a dictionary (key-value pairs). Using defaults and attempting to overwrite."
                            )
                            # Proceed using defaults, will attempt to save later
                            if not self._save_config_yaml_internal(loaded_config):
                                logger.error(
                                    f"Failed to overwrite invalid {CONFIG_FILE_PATH} with defaults."
                                )

                except yaml.YAMLError as e:
                    logger.error(
                        f"Error parsing {CONFIG_FILE_PATH}: {e}. Using defaults and attempting to overwrite."
                    )
                    if not self._save_config_yaml_internal(loaded_config):
                        logger.error(
                            f"Failed to overwrite corrupted {CONFIG_FILE_PATH} with defaults."
                        )
                except Exception as e:
                    logger.error(
                        f"Unexpected error loading {CONFIG_FILE_PATH}: {e}. Using defaults.",
                        exc_info=True,
                    )
                    # Don't try to save if it was an unexpected error

            else:
                logger.info(
                    f"{CONFIG_FILE_PATH} not found. Creating initial configuration..."
                )
                # Seed from .env overrides onto the defaults
                loaded_config = self._load_env_overrides(loaded_config)
                # Save the newly created config
                if self._save_config_yaml_internal(loaded_config):
                    logger.info(
                        f"Successfully created and saved initial configuration to {CONFIG_FILE_PATH}."
                    )
                else:
                    logger.error(
                        f"Failed to save initial configuration to {CONFIG_FILE_PATH}. Using in-memory defaults."
                    )

            self.config = loaded_config
            logger.debug(f"Current config loaded: {self.config}")
            return self.config

    def _save_config_yaml_internal(self, config_dict: Dict[str, Any]) -> bool:
        """
        Internal method to save the configuration dictionary to config.yaml.
        Includes backup and restore mechanism. Assumes lock is already held.
        """
        temp_file_path = CONFIG_FILE_PATH + ".tmp"
        backup_file_path = CONFIG_FILE_PATH + ".bak"

        try:
            # Write to temporary file first
            with open(temp_file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict, f, default_flow_style=False, sort_keys=False, indent=2
                )

            # Backup existing file if it exists
            if os.path.exists(CONFIG_FILE_PATH):
                try:
                    shutil.move(CONFIG_FILE_PATH, backup_file_path)
                    logger.debug(f"Backed up existing config to {backup_file_path}")
                except Exception as backup_err:
                    logger.warning(
                        f"Could not create backup of {CONFIG_FILE_PATH}: {backup_err}"
                    )
                    # Decide whether to proceed without backup or fail
                    # Proceeding for now, but log the warning.

            # Rename temporary file to actual config file
            shutil.move(temp_file_path, CONFIG_FILE_PATH)
            logger.info(f"Configuration successfully saved to {CONFIG_FILE_PATH}")
            return True

        except yaml.YAMLError as e:
            logger.error(
                f"Error formatting data for {CONFIG_FILE_PATH}: {e}", exc_info=True
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to save configuration to {CONFIG_FILE_PATH}: {e}",
                exc_info=True,
            )
            # Attempt to restore backup if save failed mid-way
            if os.path.exists(backup_file_path) and not os.path.exists(
                CONFIG_FILE_PATH
            ):
                try:
                    shutil.move(backup_file_path, CONFIG_FILE_PATH)
                    logger.info(
                        f"Restored configuration from backup {backup_file_path}"
                    )
                except Exception as restore_err:
                    logger.error(
                        f"Failed to restore configuration from backup: {restore_err}"
                    )
            # Clean up temp file if it still exists
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as remove_err:
                    logger.warning(
                        f"Could not remove temporary config file {temp_file_path}: {remove_err}"
                    )
            return False
        finally:
            # Clean up backup file if main file exists and save was successful (or if backup failed)
            if os.path.exists(CONFIG_FILE_PATH) and os.path.exists(backup_file_path):
                try:
                    # Only remove backup if the main file write seems okay
                    if os.path.getsize(CONFIG_FILE_PATH) > 0:  # Basic check
                        os.remove(backup_file_path)
                        logger.debug(f"Removed backup file {backup_file_path}")
                except Exception as remove_bak_err:
                    logger.warning(
                        f"Could not remove backup config file {backup_file_path}: {remove_bak_err}"
                    )

    def save_config_yaml(self, config_dict: Dict[str, Any]) -> bool:
        """Public method to save the configuration dictionary with locking."""
        with self._lock:
            return self._save_config_yaml_internal(config_dict)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-separated key path.
        e.g., get('server.port', 8000)
        """
        keys = key_path.split(".")
        value = _get_nested_value(self.config, keys, default)
        # Ensure we return a copy for mutable types like dicts/lists
        return deepcopy(value) if isinstance(value, (dict, list)) else value

    def get_all(self) -> Dict[str, Any]:
        """Get a deep copy of all current configuration values."""
        with self._lock:  # Ensure consistency while copying
            return deepcopy(self.config)

    def update_and_save(self, partial_update_dict: Dict[str, Any]) -> bool:
        """
        Deep merges a partial update dictionary into the current config
        and saves the entire configuration back to the YAML file.
        """
        if not isinstance(partial_update_dict, dict):
            logger.error("Invalid partial update data: must be a dictionary.")
            return False

        with self._lock:
            try:
                # Create a deep copy to avoid modifying self.config directly before successful save
                updated_config = deepcopy(self.config)
                # Merge the partial update into the copy
                _deep_merge_dicts(partial_update_dict, updated_config)

                # Save the fully merged configuration
                if self._save_config_yaml_internal(updated_config):
                    # If save was successful, update the in-memory config
                    self.config = updated_config
                    logger.info("Configuration updated and saved successfully.")
                    return True
                else:
                    logger.error("Failed to save updated configuration.")
                    return False
            except Exception as e:
                logger.error(
                    f"Error during configuration update and save: {e}", exc_info=True
                )
                return False

    def reset_and_save(self) -> bool:
        """
        Resets the configuration to hardcoded defaults, potentially overridden
        by values in the .env file, and saves it to config.yaml.
        """
        with self._lock:
            logger.warning(
                "Resetting configuration to defaults (with .env overrides)..."
            )
            reset_config = self._load_defaults()
            reset_config = self._load_env_overrides(
                reset_config
            )  # Apply .env overrides to defaults

            if self._save_config_yaml_internal(reset_config):
                self.config = reset_config  # Update in-memory config
                logger.info("Configuration successfully reset and saved.")
                return True
            else:
                logger.error("Failed to save reset configuration.")
                # Keep the old config in memory if save failed
                return False

    # --- Type-specific Getters with Error Handling ---
    def get_int(self, key_path: str, default: Optional[int] = None) -> int:
        """Get a configuration value as an integer."""
        value = self.get(key_path)
        if value is None:
            if default is not None:
                logger.debug(
                    f"Config '{key_path}' not found, using provided default: {default}"
                )
                return default
            else:
                logger.error(
                    f"Mandatory config '{key_path}' not found and no default. Returning 0."
                )
                return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid integer value '{value}' for '{key_path}'. Using default: {default}"
            )
            if isinstance(default, int):
                return default
            else:
                logger.error(
                    f"Cannot parse '{value}' as int for '{key_path}' and no valid default. Returning 0."
                )
                return 0

    def get_float(self, key_path: str, default: Optional[float] = None) -> float:
        """Get a configuration value as a float."""
        value = self.get(key_path)
        if value is None:
            if default is not None:
                logger.debug(
                    f"Config '{key_path}' not found, using provided default: {default}"
                )
                return default
            else:
                logger.error(
                    f"Mandatory config '{key_path}' not found and no default. Returning 0.0."
                )
                return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float value '{value}' for '{key_path}'. Using default: {default}"
            )
            if isinstance(default, float):
                return default
            else:
                logger.error(
                    f"Cannot parse '{value}' as float for '{key_path}' and no valid default. Returning 0.0."
                )
                return 0.0

    def get_bool(self, key_path: str, default: Optional[bool] = None) -> bool:
        """Get a configuration value as a boolean."""
        value = self.get(key_path)
        if value is None:
            if default is not None:
                logger.debug(
                    f"Config '{key_path}' not found, using provided default: {default}"
                )
                return default
            else:
                logger.error(
                    f"Mandatory config '{key_path}' not found and no default. Returning False."
                )
                return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "t", "yes", "y")
        try:
            # Handle numeric representations (e.g., 1 for True, 0 for False)
            return bool(int(value))
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid boolean value '{value}' for '{key_path}'. Using default: {default}"
            )
            if isinstance(default, bool):
                return default
            else:
                logger.error(
                    f"Cannot parse '{value}' as bool for '{key_path}' and no valid default. Returning False."
                )
                return False


# --- Create a singleton instance for global access ---
config_manager = YamlConfigManager()

# --- Export common getters for easy access ---


# Helper to get default value from the DEFAULT_CONFIG structure
def _get_default(key_path: str) -> Any:
    keys = key_path.split(".")
    return _get_nested_value(DEFAULT_CONFIG, keys)


# Server Settings
def get_host() -> str:
    return config_manager.get("server.host", _get_default("server.host"))


def get_port() -> int:
    return config_manager.get_int("server.port", _get_default("server.port"))


# Model Source Settings
def get_model_repo_id() -> str:
    return config_manager.get("model.repo_id", _get_default("model.repo_id"))


def get_model_config_filename() -> str:
    return config_manager.get(
        "model.config_filename", _get_default("model.config_filename")
    )


def get_model_weights_filename() -> str:
    return config_manager.get(
        "model.weights_filename", _get_default("model.weights_filename")
    )


def get_whisper_model_name() -> str:
    return config_manager.get(
        "model.whisper_model_name", _get_default("model.whisper_model_name")
    )


# Path Settings
def get_model_cache_path() -> str:
    return os.path.abspath(
        config_manager.get("paths.model_cache", _get_default("paths.model_cache"))
    )


def get_reference_audio_path() -> str:
    return os.path.abspath(
        config_manager.get(
            "paths.reference_audio", _get_default("paths.reference_audio")
        )
    )


def get_output_path() -> str:
    return os.path.abspath(
        config_manager.get("paths.output", _get_default("paths.output"))
    )


def get_predefined_voices_path() -> str:
    return os.path.abspath(
        config_manager.get("paths.voices", _get_default("paths.voices"))
    )


# Default Generation Parameter Getters
def get_gen_default_speed_factor() -> float:
    return config_manager.get_float(
        "generation_defaults.speed_factor",
        _get_default("generation_defaults.speed_factor"),
    )


def get_gen_default_cfg_scale() -> float:
    return config_manager.get_float(
        "generation_defaults.cfg_scale", _get_default("generation_defaults.cfg_scale")
    )


def get_gen_default_temperature() -> float:
    return config_manager.get_float(
        "generation_defaults.temperature",
        _get_default("generation_defaults.temperature"),
    )


def get_gen_default_top_p() -> float:
    return config_manager.get_float(
        "generation_defaults.top_p", _get_default("generation_defaults.top_p")
    )


def get_gen_default_cfg_filter_top_k() -> int:
    return config_manager.get_int(
        "generation_defaults.cfg_filter_top_k",
        _get_default("generation_defaults.cfg_filter_top_k"),
    )


def get_gen_default_seed() -> int:
    return config_manager.get_int(
        "generation_defaults.seed", _get_default("generation_defaults.seed")
    )


def get_gen_default_split_text() -> bool:
    return config_manager.get_bool(
        "generation_defaults.split_text", _get_default("generation_defaults.split_text")
    )


def get_gen_default_chunk_size() -> int:
    return config_manager.get_int(
        "generation_defaults.chunk_size", _get_default("generation_defaults.chunk_size")
    )


# UI State Getters (might be less frequently needed directly in backend)
def get_ui_state() -> Dict[str, Any]:
    """Gets the entire UI state dictionary."""
    return config_manager.get("ui_state", _get_default("ui_state"))


def get_hide_chunk_warning() -> bool:
    """Gets the flag for hiding the chunk warning dialog."""
    return config_manager.get_bool(
        "ui_state.hide_chunk_warning", _get_default("ui_state.hide_chunk_warning")
    )


def get_hide_generation_warning() -> bool:
    """Gets the flag for hiding the general generation warning dialog."""
    return config_manager.get_bool(
        "ui_state.hide_generation_warning",
        _get_default("ui_state.hide_generation_warning"),
    )
