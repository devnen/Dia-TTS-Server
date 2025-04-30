# utils.py
# Utility functions for the Dia TTS server

import logging
import time
import os
import uuid
import io
import re
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Dict, Any, Set, List

# Import config manager to get paths dynamically
from config import config_manager, get_predefined_voices_path, get_reference_audio_path

# Optional import for unvoiced segment detection
try:
    import parselmouth

    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logging.warning(
        "Parselmouth not installed. Unvoiced segment removal will be disabled."
    )

# Optional import for Whisper transcription
try:
    import whisper

    WHISPER_AVAILABLE_UTIL = True
except ImportError:
    WHISPER_AVAILABLE_UTIL = False
    # Warning handled during actual usage attempt

logger = logging.getLogger(__name__)

# --- Filename Sanitization ---


def sanitize_filename(filename: str) -> str:
    """
    Removes potentially unsafe characters and path components from a filename
    to make it safe for use in file paths. Replaces unsafe sequences with single underscore.

    Args:
        filename: The original filename string.

    Returns:
        A sanitized filename string.
    """
    if not filename:
        return f"empty_filename_{uuid.uuid4().hex[:8]}"

    # Remove directory separators and leading/trailing whitespace
    filename = os.path.basename(filename).strip()
    if not filename:
        return f"empty_filename_{uuid.uuid4().hex[:8]}"

    # Keep only alphanumeric, underscore, hyphen, dot, space.
    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
    )
    sanitized_list = []
    last_char_was_underscore = False

    for char in filename:
        if char in safe_chars:
            sanitized_list.append(char)
            last_char_was_underscore = False
        elif not last_char_was_underscore:
            sanitized_list.append("_")
            last_char_was_underscore = True

    sanitized = "".join(sanitized_list).strip("_")

    # Prevent names starting with dot or consisting only of dots/spaces/underscores
    if not sanitized or sanitized.lstrip("._ ") == "":
        return f"uploaded_file_{uuid.uuid4().hex[:8]}"

    # Limit length (e.g., 100 chars), preserving extension
    max_len = 100
    if len(sanitized) > max_len:
        name, ext = os.path.splitext(sanitized)
        name = name[: max_len - len(ext)] if len(name) > (max_len - len(ext)) else name
        sanitized = name + ext
        logger.warning(
            f"Original filename '{filename}' truncated to '{sanitized}' due to length."
        )

    if not sanitized:
        return f"sanitized_empty_{uuid.uuid4().hex[:8]}"

    return sanitized


# --- Constants for Text Processing ---
ABBREVIATIONS: Set[str] = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "rev.",
    "hon.",
    "st.",
    "etc.",
    "e.g.",
    "i.e.",
    "vs.",
    "approx.",
    "apt.",
    "dept.",
    "fig.",
    "gen.",
    "gov.",
    "inc.",
    "jr.",
    "sr.",
    "ltd.",
    "no.",
    "p.",
    "pp.",
    "vol.",
    "op.",
    "cit.",
    "ca.",
    "cf.",
    "ed.",
    "esp.",
    "et.",
    "al.",
    "ibid.",
    "id.",
    "inf.",
    "sup.",
    "viz.",
    "sc.",
    "fl.",
    "d.",
    "b.",
    "r.",
    "c.",
    "v.",
    "u.s.",
    "u.k.",
    "a.m.",
    "p.m.",
    "a.d.",
    "b.c.",
}

# Common titles that might appear without a period if cleaned beforehand
TITLES_NO_PERIOD: Set[str] = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "rev",
    "hon",
    "st",
    "sgt",
    "capt",
    "lt",
    "col",
    "gen",
}

# Regex patterns (pre-compile for efficiency)
NUMBER_DOT_NUMBER_PATTERN = re.compile(r"(?<!\d\.)\d*\.\d+")
VERSION_PATTERN = re.compile(r"[vV]?\d+(\.\d+)+")
# Pattern to find potential sentence endings (punctuation followed by space or end)
POTENTIAL_END_PATTERN = re.compile(r'([.!?])(["\']?)(\s+|$)')
# Pattern to find speaker tags like [S1], [S2], etc.
SPEAKER_TAG_PATTERN = re.compile(r"(\[S\d+\])")
# Default speaker tag if none is specified at the beginning
DEFAULT_SPEAKER_TAG = "[S1]"
# Pattern to detect start-of-line bullet points (updated)
BULLET_POINT_PATTERN = re.compile(r"(?:^|\n)\s*([-•*]|\d+\.)\s+")

# --- Audio Processing ---


def encode_audio(
    audio_array: np.ndarray, sample_rate: int, output_format: str = "opus"
) -> Optional[bytes]:
    """
    Encodes a NumPy audio array into the specified format in memory.

    Args:
        audio_array: NumPy array containing audio data (float32, range [-1, 1]).
        sample_rate: Sample rate of the audio data.
        output_format: Desired output format ('opus' or 'wav').

    Returns:
        Bytes object containing the encoded audio, or None on failure.
    """
    if audio_array is None or audio_array.size == 0:
        logger.warning("encode_audio received empty or None audio array.")
        return None

    start_time = time.time()
    output_buffer = io.BytesIO()

    try:
        if output_format == "opus":
            # Soundfile handles float32 for Opus correctly with format='ogg', subtype='opus'
            sf.write(
                output_buffer, audio_array, sample_rate, format="ogg", subtype="opus"
            )
            # content_type = "audio/ogg; codecs=opus" # More specific
            content_type = "audio/opus"  # Match OpenAI response type
        elif output_format == "wav":
            # WAV typically uses int16 for broader compatibility
            # Ensure clipping doesn't occur if input slightly exceeds [-1, 1]
            audio_clipped = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            sf.write(
                output_buffer, audio_int16, sample_rate, format="wav", subtype="pcm_16"
            )
            content_type = "audio/wav"
        else:
            logger.error(f"Unsupported output format requested: {output_format}")
            return None

        encoded_bytes = output_buffer.getvalue()
        end_time = time.time()
        logger.info(
            f"Encoded {len(encoded_bytes)} bytes to {output_format} in {end_time - start_time:.3f} seconds."
        )
        return encoded_bytes

    except ImportError:
        logger.critical(
            "`soundfile` or its dependency `libsndfile` not found/installed correctly. Cannot encode audio."
        )
        # Depending on server setup, might want to raise or return None
        # raise # Re-raise critical error if server should stop
        return None  # Return None to allow server to handle gracefully
    except Exception as e:
        logger.error(f"Error encoding audio to {output_format}: {e}", exc_info=True)
        return None


def save_audio_to_file(
    audio_array: np.ndarray, sample_rate: int, file_path: str
) -> bool:
    """
    Saves a NumPy audio array to a WAV file.

    Args:
        audio_array: NumPy array containing audio data (float32, range [-1, 1]).
        sample_rate: Sample rate of the audio data.
        file_path: Path to save the WAV file.

    Returns:
        True if saving was successful, False otherwise.
    """
    if audio_array is None or audio_array.size == 0:
        logger.warning("save_audio_to_file received empty or None audio array.")
        return False
    if not file_path.lower().endswith(".wav"):
        original_path = file_path
        file_path += ".wav"
        logger.warning(
            f"File path '{original_path}' did not end with .wav. Saving as '{file_path}'."
        )

    start_time = time.time()
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir:  # Avoid error if saving to current directory
            os.makedirs(output_dir, exist_ok=True)

        # WAV typically uses int16 for broader compatibility
        # Ensure clipping doesn't occur if input slightly exceeds [-1, 1]
        audio_clipped = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        sf.write(file_path, audio_int16, sample_rate, format="wav", subtype="pcm_16")

        end_time = time.time()
        logger.info(
            f"Saved WAV file to {file_path} in {end_time - start_time:.3f} seconds."
        )
        return True
    except ImportError:
        logger.critical(
            "`soundfile` or its dependency `libsndfile` not found/installed correctly. Cannot save audio."
        )
        return False  # Indicate failure
    except Exception as e:
        logger.error(f"Error saving WAV file to {file_path}: {e}", exc_info=True)
        return False


def trim_lead_trail_silence(
    audio_array: np.ndarray,
    silence_threshold: float = 0.01,
    min_silence_duration_ms: int = 500,
    padding_ms: int = 200,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Trims silence from beginning and end of an audio array."""
    if audio_array.size == 0:
        return audio_array

    # Calculate sample positions
    samples_per_ms = sample_rate / 1000
    min_silence_samples = int(min_silence_duration_ms * samples_per_ms)
    padding_samples = int(padding_ms * samples_per_ms)

    # Find leading silence
    amplitude = np.abs(audio_array)
    is_silence = amplitude < silence_threshold

    # Find first non-silent sample
    leading_silence = 0
    for i in range(len(is_silence)):
        if not is_silence[i]:
            leading_silence = i
            break
    else:  # Handle case where entire audio is silent
        return np.array([], dtype=audio_array.dtype)

    # Find trailing silence
    trailing_silence = len(audio_array)
    for i in range(len(is_silence) - 1, -1, -1):
        if not is_silence[i]:
            trailing_silence = i + 1
            break

    # Check if silence exceeds thresholds
    needs_trim = (
        leading_silence > min_silence_samples
        or len(audio_array) - trailing_silence > min_silence_samples
    )
    if not needs_trim:
        return audio_array

    # Ensure we don't go out of bounds
    start = max(0, leading_silence - padding_samples)
    end = min(len(audio_array), trailing_silence + padding_samples)

    # Return trimmed audio
    return audio_array[start:end]


def fix_internal_silence(
    audio_array: np.ndarray,
    silence_threshold: float = 0.01,
    min_silence_ms: int = 1000,
    max_silence_ms: int = 500,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Reduces long internal silences to a specified maximum duration."""
    if audio_array.size == 0:
        return audio_array

    # Calculate sample positions
    samples_per_ms = sample_rate / 1000
    min_silence_samples = int(min_silence_ms * samples_per_ms)
    max_silence_samples = int(max_silence_ms * samples_per_ms)

    # Find silent regions
    amplitude = np.abs(audio_array)
    is_silence = amplitude < silence_threshold

    # Find silent segments
    silent_starts = []
    silent_ends = []
    in_silence = False
    curr_start = 0

    for i in range(len(is_silence)):
        if is_silence[i] and not in_silence:
            # Start of silence
            in_silence = True
            curr_start = i
        elif not is_silence[i] and in_silence:
            # End of silence
            in_silence = False
            if i - curr_start >= min_silence_samples:
                silent_starts.append(curr_start)
                silent_ends.append(i)

    # Handle if audio ends in silence
    if in_silence and len(audio_array) - curr_start >= min_silence_samples:
        silent_starts.append(curr_start)
        silent_ends.append(len(audio_array))

    # If no long silences found, return original
    if not silent_starts:
        return audio_array
    result, last_end = [], 0
    for start, end in zip(silent_starts, silent_ends):
        result.append(audio_array[last_end:start])
        silence_to_keep = audio_array[start : min(end, start + max_silence_samples)]
        result.append(silence_to_keep)
        last_end = end
    if last_end < len(audio_array):
        result.append(audio_array[last_end:])
    if not result:
        return np.array([], dtype=audio_array.dtype)
    return np.concatenate(result)


def remove_long_unvoiced_segments(
    audio_array: np.ndarray,
    min_unvoiced_duration_ms: int = 300,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Removes segments that are unvoiced for longer than the specified duration."""
    if not PARSELMOUTH_AVAILABLE:
        logger.warning("Parselmouth not available, skipping unvoiced segment removal")
        return audio_array
    if audio_array.size == 0:
        return audio_array
    try:
        samples = audio_array.astype(np.float64)
        sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
        pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        frame_times = pitch.xs()
        num_frames = len(frame_times)
        if num_frames == 0:
            logger.warning("Could not extract pitch frames, skipping unvoiced removal.")
            return audio_array
        is_voiced = np.zeros(num_frames, dtype=bool)
        for i in range(num_frames):
            f0 = pitch.get_value_at_time(frame_times[i])
            is_voiced[i] = f0 > 0 and not np.isnan(f0)
        is_unvoiced = ~is_voiced
        diffs = np.diff(np.concatenate(([False], is_unvoiced, [False])).astype(int))
        unvoiced_starts = np.where(diffs == 1)[0]
        unvoiced_ends = np.where(diffs == -1)[0]
        frame_duration = (
            frame_times[1] - frame_times[0]
            if len(frame_times) > 1
            else (sound.duration / num_frames if num_frames > 0 else 0)
        )
        min_unvoiced_duration_s = min_unvoiced_duration_ms / 1000.0
        segments_to_keep = []
        last_keep_end = 0.0
        for start_idx, end_idx in zip(unvoiced_starts, unvoiced_ends):
            if start_idx >= num_frames or end_idx > num_frames:
                continue
            start_time = (
                frame_times[start_idx]
                if start_idx < len(frame_times)
                else start_idx * frame_duration
            )
            end_time = (
                frame_times[end_idx - 1]
                if end_idx - 1 < len(frame_times)
                else (end_idx - 1) * frame_duration
            )
            duration = end_time - start_time
            if start_time > last_keep_end:
                start_sample, end_sample = int(last_keep_end * sample_rate), int(
                    start_time * sample_rate
                )
                if end_sample > start_sample:
                    segments_to_keep.append((start_sample, end_sample))
            if duration >= min_unvoiced_duration_s:
                last_keep_end = end_time
            else:
                start_sample, end_sample = int(start_time * sample_rate), int(
                    end_time * sample_rate
                )
                if end_sample > start_sample:
                    segments_to_keep.append((start_sample, end_sample))
                last_keep_end = end_time
        if last_keep_end < sound.duration:
            start_sample, end_sample = int(last_keep_end * sample_rate), len(
                audio_array
            )
            if end_sample > start_sample:
                segments_to_keep.append((start_sample, end_sample))
        if not segments_to_keep:
            logger.warning(
                "Unvoiced segment removal resulted in empty audio, returning original."
            )
            return audio_array
        if (
            len(segments_to_keep) == 1
            and segments_to_keep[0][0] == 0
            and segments_to_keep[0][1] >= len(audio_array) - 1
        ):
            return audio_array
        result = []
        for start, end in segments_to_keep:
            start, end = max(0, start), min(len(audio_array), end)
            if end > start:
                result.append(audio_array[start:end])
        if not result:
            logger.warning(
                "Unvoiced segment removal concatenation resulted in empty audio, returning original."
            )
            return audio_array
        return np.concatenate(result)
    except Exception as e:
        logger.error(f"Error in unvoiced segment removal: {e}", exc_info=True)
        return audio_array


# --- Text Processing ---


def _is_valid_sentence_end(text: str, index: int) -> bool:
    """Checks if a period at a given index is a valid sentence end."""
    start_word_before = index - 1
    scan_limit = max(0, index - 10)
    while start_word_before >= scan_limit and not text[start_word_before].isspace():
        start_word_before -= 1
    word_before = text[start_word_before + 1 : index + 1].lower()
    if word_before in ABBREVIATIONS:
        return False
    search_start, search_end = max(0, index - 10), min(len(text), index + 10)
    context_segment = text[search_start:search_end]
    relative_dot_index = index - search_start
    for pattern in [NUMBER_DOT_NUMBER_PATTERN, VERSION_PATTERN]:
        for match in pattern.finditer(context_segment):
            if match.start() <= relative_dot_index < match.end():
                is_last_char_of_match = relative_dot_index == match.end() - 1
                is_followed_by_space_or_end = (
                    index + 1 == len(text) or text[index + 1].isspace()
                )
                if not (is_last_char_of_match and is_followed_by_space_or_end):
                    return False
    return True


def _split_text_by_punctuation(text: str) -> List[str]:
    """Splits text into sentences based on punctuation marks (.!?)."""
    sentences, last_split, text_len = [], 0, len(text)
    potential_ends = POTENTIAL_END_PATTERN.finditer(text)
    for match in potential_ends:
        punct_index, punct_char = match.start(1), text[match.start(1)]
        quote_match = match.group(2)
        quote_len = len(quote_match) if quote_match is not None else 0
        slice_end_index = match.start(1) + 1 + quote_len
        if punct_char in ["!", "?"]:
            current_sentence = text[last_split:slice_end_index].strip()
            if current_sentence:
                sentences.append(current_sentence)
            last_split = match.end()
            continue
        if punct_char == ".":
            if (punct_index > 0 and text[punct_index - 1] == ".") or (
                punct_index < text_len - 1 and text[punct_index + 1] == "."
            ):
                continue
            if _is_valid_sentence_end(text, punct_index):
                current_sentence = text[last_split:slice_end_index].strip()
                if current_sentence:
                    sentences.append(current_sentence)
                last_split = match.end()
    remaining_text = text[last_split:].strip()
    if remaining_text:
        sentences.append(remaining_text)
    sentences = [s for s in sentences if s]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def split_into_sentences(text: str) -> List[str]:
    """Splits tag-free text into sentences (or bullet points)."""
    if not text or text.isspace():
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    bullet_matches = list(BULLET_POINT_PATTERN.finditer(text))
    if bullet_matches:
        logger.debug("Bullet points detected, splitting accordingly.")
        sentences, start_pos = [], 0
        for i, match in enumerate(bullet_matches):
            bullet_start_char_index = match.start()
            if i == 0 and bullet_start_char_index > start_pos:
                pre_bullet_text = text[start_pos:bullet_start_char_index].strip()
                if pre_bullet_text:
                    sentences.extend(
                        s for s in _split_text_by_punctuation(pre_bullet_text) if s
                    )
            bullet_end_char_index = (
                bullet_matches[i + 1].start()
                if i + 1 < len(bullet_matches)
                else len(text)
            )
            bullet_item_text = text[
                bullet_start_char_index:bullet_end_char_index
            ].strip()
            if bullet_item_text:
                sentences.append(bullet_item_text)
            start_pos = bullet_end_char_index
        if start_pos < len(text):
            post_bullet_text = text[start_pos:].strip()
            if post_bullet_text:
                sentences.extend(
                    s for s in _split_text_by_punctuation(post_bullet_text) if s
                )
        return [s for s in sentences if s]
    else:
        logger.debug("No bullet points detected, using punctuation splitting.")
        return _split_text_by_punctuation(text)


def _preprocess_and_tag_sentences(full_text: str) -> List[Tuple[str, str]]:
    """Pre-processes text to associate speaker tags with sentences."""
    if not full_text or full_text.isspace():
        return []
    tagged_sentences, segments = [], SPEAKER_TAG_PATTERN.split(full_text)
    current_tag = DEFAULT_SPEAKER_TAG
    if segments and SPEAKER_TAG_PATTERN.fullmatch(segments[0]):
        current_tag, segments = segments[0], segments[1:]
    elif (
        segments
        and segments[0] == ""
        and len(segments) > 1
        and SPEAKER_TAG_PATTERN.fullmatch(segments[1])
    ):
        current_tag, segments = segments[1], segments[2:]
    elif (
        segments
        and segments[0].isspace()
        and len(segments) > 1
        and SPEAKER_TAG_PATTERN.fullmatch(segments[1])
    ):
        current_tag, segments = segments[1], segments[2:]
    buffer = ""
    for segment in segments:
        if not segment:
            continue
        if SPEAKER_TAG_PATTERN.fullmatch(segment):
            if buffer.strip():
                sentences_in_buffer = split_into_sentences(buffer.strip())
                for sentence in sentences_in_buffer:
                    if sentence:
                        tagged_sentences.append((current_tag, sentence))
            current_tag, buffer = segment, ""
        else:
            buffer += segment
    if buffer.strip():
        sentences_in_buffer = split_into_sentences(buffer.strip())
        for sentence in sentences_in_buffer:
            if sentence:
                tagged_sentences.append((current_tag, sentence))
    if not tagged_sentences and full_text.strip():
        cleaned_full_text = full_text.strip()
        match = SPEAKER_TAG_PATTERN.match(cleaned_full_text)
        start_tag = match.group(1) if match else DEFAULT_SPEAKER_TAG
        content_after_tag = SPEAKER_TAG_PATTERN.sub(
            "", cleaned_full_text, count=1
        ).strip()
        if content_after_tag:
            tagged_sentences.append((start_tag, content_after_tag))
        elif start_tag != DEFAULT_SPEAKER_TAG:
            logger.warning(
                f"Input text contains tag '{start_tag}' but no subsequent content."
            )
    logger.debug(f"Preprocessed into {len(tagged_sentences)} tagged sentences.")
    return tagged_sentences


def chunk_text_by_sentences(
    full_text: str,
    chunk_size: int,
    allow_multiple_tags: bool = False,
) -> List[str]:
    """Chunks text based on sentences and speaker tags."""
    if not full_text or full_text.isspace():
        return []
    if chunk_size <= 0:
        chunk_size = float("inf")
    tagged_sentences = _preprocess_and_tag_sentences(full_text)
    if not tagged_sentences:
        return []
    text_chunks, current_chunk_sentences, current_chunk_tag, current_chunk_len = (
        [],
        [],
        None,
        0,
    )
    for i, (sentence_tag, sentence_text) in enumerate(tagged_sentences):
        sentence_len = len(sentence_text)
        start_new_chunk = False
        if current_chunk_tag is None:
            start_new_chunk = True
        elif not allow_multiple_tags and sentence_tag != current_chunk_tag:
            logger.debug(
                f"Forcing new chunk due to tag change: {current_chunk_tag} -> {sentence_tag}"
            )
            start_new_chunk = True
        if start_new_chunk:
            if current_chunk_sentences:
                formatted_chunk = (
                    f"{current_chunk_tag} {' '.join(current_chunk_sentences)}"
                )
                text_chunks.append(formatted_chunk)
                logger.debug(
                    f"Finalized chunk (tag change/first): {formatted_chunk[:80]}..."
                )
            current_chunk_sentences, current_chunk_tag, current_chunk_len = (
                [sentence_text],
                sentence_tag,
                sentence_len,
            )
            if sentence_len > chunk_size:
                logger.info(
                    f"Single sentence (speaker {sentence_tag}, length {sentence_len}) exceeds chunk_size {chunk_size}. Keeping it intact."
                )
                formatted_chunk = (
                    f"{current_chunk_tag} {' '.join(current_chunk_sentences)}"
                )
                text_chunks.append(formatted_chunk)
                logger.debug(
                    f"Finalized chunk (long sentence): {formatted_chunk[:80]}..."
                )
                current_chunk_sentences, current_chunk_tag, current_chunk_len = (
                    [],
                    None,
                    0,
                )
        else:
            potential_len = current_chunk_len + 1 + sentence_len
            if potential_len <= chunk_size:
                current_chunk_sentences.append(sentence_text)
                current_chunk_len = potential_len
                if allow_multiple_tags:
                    current_chunk_tag = sentence_tag
                logger.debug(
                    f"Added sentence to current chunk (tag {current_chunk_tag}). New length: {current_chunk_len}"
                )
            else:
                if current_chunk_sentences:
                    formatted_chunk = (
                        f"{current_chunk_tag} {' '.join(current_chunk_sentences)}"
                    )
                    text_chunks.append(formatted_chunk)
                    logger.debug(
                        f"Finalized chunk (size limit): {formatted_chunk[:80]}..."
                    )
                current_chunk_sentences, current_chunk_tag, current_chunk_len = (
                    [sentence_text],
                    sentence_tag,
                    sentence_len,
                )
                logger.debug(
                    f"Started new chunk with sentence (tag {sentence_tag}). Length: {current_chunk_len}"
                )
                if sentence_len > chunk_size:
                    logger.info(
                        f"Single sentence (speaker {sentence_tag}, length {sentence_len}) exceeds chunk_size {chunk_size}. Keeping it intact."
                    )
                    formatted_chunk = (
                        f"{current_chunk_tag} {' '.join(current_chunk_sentences)}"
                    )
                    text_chunks.append(formatted_chunk)
                    logger.debug(
                        f"Finalized chunk (long sentence): {formatted_chunk[:80]}..."
                    )
                    current_chunk_sentences, current_chunk_tag, current_chunk_len = (
                        [],
                        None,
                        0,
                    )
    if current_chunk_sentences:
        formatted_chunk = f"{current_chunk_tag} {' '.join(current_chunk_sentences)}"
        text_chunks.append(formatted_chunk)
        logger.debug(f"Finalized last chunk: {formatted_chunk[:80]}...")
    if not text_chunks and full_text.strip():
        logger.warning(
            "Chunking process resulted in zero chunks despite non-empty input. Using fallback (single chunk)."
        )
        first_tag = tagged_sentences[0][0] if tagged_sentences else DEFAULT_SPEAKER_TAG
        clean_text = " ".join(sent for _, sent in tagged_sentences)
        if not clean_text:
            clean_text = SPEAKER_TAG_PATTERN.sub("", full_text).strip()
        if clean_text:
            text_chunks.append(f"{first_tag} {clean_text}")
    logger.info(f"Chunking complete. Generated {len(text_chunks)} chunks.")
    return text_chunks


# --- Whisper Transcription Utility ---


def _generate_transcript_with_whisper(
    audio_data_np: np.ndarray, model_name: str, cache_path: str
) -> Optional[str]:
    """Generates a transcript for the given audio data using Whisper."""
    if not WHISPER_AVAILABLE_UTIL:
        logger.error("Whisper library is not installed. Cannot generate transcript.")
        return None
    logger.info(f"Attempting transcription with Whisper model '{model_name}'...")
    start_time = time.time()
    try:
        model = whisper.load_model(model_name, download_root=cache_path)
        logger.info(f"Whisper model '{model_name}' loaded.")
        if audio_data_np.dtype != np.float32:
            logger.warning(
                f"Audio data for Whisper is not float32 ({audio_data_np.dtype}), converting."
            )
            audio_data_np = audio_data_np.astype(np.float32)
        result = model.transcribe(audio_data_np, language="en")
        transcript = result["text"]
        end_time = time.time()
        logger.info(
            f"Whisper transcription successful in {end_time - start_time:.2f} seconds."
        )
        logger.debug(f"Whisper transcript (raw): {transcript[:100]}...")
        return transcript
    except FileNotFoundError:
        logger.error(
            f"Whisper model '{model_name}' not found in cache '{cache_path}' or download failed."
        )
        return None
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
        return None


# --- File Listing Utilities ---


def get_valid_reference_files() -> List[str]:
    """Gets a sorted list of valid audio files (.wav, .mp3) from the reference directory."""
    ref_path = get_reference_audio_path()  # Get path from config
    valid_files = []
    allowed_extensions = (".wav", ".mp3")
    try:
        if os.path.isdir(ref_path):
            for filename in os.listdir(ref_path):
                if not filename.startswith("."):  # Ignore hidden files
                    full_path = os.path.join(ref_path, filename)
                    if os.path.isfile(full_path) and filename.lower().endswith(
                        allowed_extensions
                    ):
                        valid_files.append(filename)
        else:
            logger.warning(
                f"Reference audio directory not found: {ref_path}. Creating it."
            )
            os.makedirs(ref_path, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Error reading reference audio directory '{ref_path}': {e}", exc_info=True
        )
    return sorted(valid_files)


def get_predefined_voices() -> List[Dict[str, str]]:
    """
    Gets a sorted list of predefined voices from the voices directory.
    Formats names (e.g., "Abigail-Taylor") and handles duplicates.

    Returns:
        List of dictionaries: [{"display_name": "Formatted Name", "filename": "original_file.wav"}, ...]
    """
    voices_path = get_predefined_voices_path()  # Get path from config
    predefined_voices = []
    allowed_extensions = (
        ".wav",
    )  # Only allow wav for predefined? Or mp3 too? Assuming WAV for now.

    try:
        if not os.path.isdir(voices_path):
            logger.warning(
                f"Predefined voices directory not found: {voices_path}. Creating it."
            )
            os.makedirs(voices_path, exist_ok=True)
            return []  # Return empty list if directory was missing

        display_name_counts = {}
        temp_list = []

        for filename in os.listdir(voices_path):
            if (
                not filename.startswith(".")
                and os.path.isfile(os.path.join(voices_path, filename))
                and filename.lower().endswith(allowed_extensions)
            ):
                base_name, _ = os.path.splitext(filename)
                # Format display name: replace underscores with hyphens
                display_name_base = base_name.replace("_", "-")
                temp_list.append(
                    {
                        "original_filename": filename,
                        "display_name_base": display_name_base,
                    }
                )

        # Sort temporarily by base display name to handle duplicates consistently
        temp_list.sort(key=lambda x: x["display_name_base"])

        # Process sorted list to handle duplicates
        for item in temp_list:
            base_name = item["display_name_base"]
            original_filename = item["original_filename"]

            if base_name in display_name_counts:
                display_name_counts[base_name] += 1
                display_name = f"{base_name} ({display_name_counts[base_name]})"
            else:
                display_name_counts[base_name] = 1
                display_name = base_name

            predefined_voices.append(
                {"display_name": display_name, "filename": original_filename}
            )

        # Final sort by the generated display name
        predefined_voices.sort(key=lambda x: x["display_name"])
        logger.info(
            f"Found {len(predefined_voices)} predefined voices in {voices_path}"
        )

    except Exception as e:
        logger.error(
            f"Error reading predefined voices directory '{voices_path}': {e}",
            exc_info=True,
        )
        predefined_voices = []  # Return empty list on error

    return predefined_voices


# --- Other Utilities ---


class PerformanceMonitor:
    """Simple performance monitoring helper class."""

    def __init__(self):
        self.start_time = time.monotonic()
        self.events = []

    def record(self, event_name: str):
        self.events.append((event_name, time.monotonic()))

    def report(self) -> str:
        report_lines = ["Performance Report:"]
        last_time = self.start_time
        total_duration = time.monotonic() - self.start_time
        for name, timestamp in self.events:
            duration = timestamp - last_time
            report_lines.append(f"  - {name}: {duration:.3f}s")
            last_time = timestamp
        report_lines.append(f"Total Duration: {total_duration:.3f}s")
        return "\n".join(report_lines)
