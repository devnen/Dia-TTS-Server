# models.py
# Pydantic models for API requests and potentially responses

from pydantic import BaseModel, Field
from typing import Optional, Literal

# --- Request Models ---


class OpenAITTSRequest(BaseModel):
    """Request model compatible with the OpenAI TTS API."""

    model: str = Field(
        default="dia-1.6b",
        description="Model identifier (ignored by this server, always uses Dia). Included for compatibility.",
    )
    input: str = Field(..., description="The text to synthesize.")
    voice: str = Field(
        default="S1",
        description="Voice mode or reference audio filename. Examples: 'S1', 'S2', 'dialogue', 'my_reference.wav'.",
    )
    response_format: Literal["opus", "wav"] = Field(
        default="opus", description="The desired audio output format."
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,  # Allow wider range for speed factor post-processing
        le=2.0,
        description="Adjusts the speed of the generated audio (0.5 to 2.0).",
    )
    # Add seed parameter, defaulting to random (-1)
    seed: Optional[int] = Field(
        default=-1,
        description="Generation seed. Use -1 for random (default), or a specific integer for deterministic output.",
    )


class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint."""

    text: str = Field(
        ...,
        description="The text to synthesize. For 'dialogue' mode, include [S1]/[S2] tags.",
    )
    voice_mode: Literal["dialogue", "single_s1", "single_s2", "clone"] = Field(
        default="single_s1", description="Specifies the generation mode."
    )
    clone_reference_filename: Optional[str] = Field(
        default=None,
        description="Filename of the reference audio within the configured reference path (required if voice_mode is 'clone').",
    )
    # New: Optional transcript for cloning
    transcript: Optional[str] = Field(
        default=None,
        description="Optional transcript of the reference audio for cloning. If provided, overrides local .txt file lookup and Whisper generation.",
    )
    output_format: Literal["opus", "wav"] = Field(
        default="opus", description="The desired audio output format."
    )
    # Dia-specific generation parameters
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of audio tokens to generate per chunk (defaults to model's internal config value).",
    )
    cfg_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Classifier-Free Guidance scale (1.0-5.0).",
    )
    temperature: float = Field(
        default=1.3,
        ge=0.1,
        le=1.5,
        description="Sampling temperature (0.1-1.5).",  # Allow lower temp for greedy-like
    )
    top_p: float = Field(
        default=0.95,
        ge=0.1,  # Allow lower top_p
        le=1.0,
        description="Nucleus sampling probability (0.1-1.0).",
    )
    speed_factor: float = Field(
        default=0.94,
        ge=0.5,  # Allow wider range for speed factor post-processing
        le=2.0,
        description="Adjusts the speed of the generated audio (0.5 to 2.0).",
    )
    cfg_filter_top_k: int = Field(
        default=35,
        ge=1,
        le=100,
        description="Top k filter for CFG guidance (1-100).",  # Allow wider range
    )
    # Add seed parameter, defaulting to random (-1)
    seed: Optional[int] = Field(
        default=-1,
        description="Generation seed. Use -1 for random (default), or a specific integer for deterministic output.",
    )
    # Add text splitting parameters
    split_text: Optional[bool] = Field(
        default=True,  # Default to splitting enabled
        description="Whether to automatically split long text into chunks for processing.",
    )
    chunk_size: Optional[int] = Field(
        default=300,  # Default target chunk size
        ge=100,  # Minimum reasonable chunk size
        le=1000,  # Maximum reasonable chunk size
        description="Approximate target character length for text chunks when splitting is enabled (100-1000).",
    )


# --- Response Models (Optional, can be simple dicts too) ---


class TTSResponse(BaseModel):
    """Basic response model for successful generation (if returning JSON)."""

    request_id: str
    status: str = "completed"
    generation_time_sec: float
    output_url: Optional[str] = None  # If saving file and returning URL


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str
