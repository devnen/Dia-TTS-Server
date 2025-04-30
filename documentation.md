# Dia TTS Server - Technical Documentation

**Version:** 1.4.0
**Date:** 2025-04-29

**Table of Contents:**

1.  [Overview](#1-overview)
2.  [What's New (v1.4.0)](#2-whats-new-v140)
3.  [Large Text Processing & Chunking: Generating Long Audio Naturally](#3-large-text-processing--chunking-generating-long-audio-naturally)
    *   [3.1 The Challenge: Why Long Text is Hard for TTS](#31-the-challenge-why-long-text-is-hard-for-tts)
    *   [3.2 Our Solution: Intelligent, Context-Aware Chunking](#32-our-solution-intelligent-context-aware-chunking)
    *   [3.3 The Chunking Process: A Step-by-Step Look](#33-the-chunking-process-a-step-by-step-look)
        *   [Step 1: Understanding the Structure - Tagging and Sentence Splitting](#step-1-understanding-the-structure---tagging-and-sentence-splitting)
        *   [Step 2: Building the Chunks - Aggregation with Rules](#step-2-building-the-chunks---aggregation-with-rules)
        *   [Step 3: Final Touches - Formatting the Output](#step-3-final-touches---formatting-the-output)
        *   [Example Walkthrough](#example-walkthrough)
    *   [3.4 The `chunk_size` Parameter: Controlling Segment Length](#34-the-chunk_size-parameter-controlling-segment-length)
    *   [3.5 Why This Matters: Benefits of Smart Chunking](#35-why-this-matters-benefits-of-smart-chunking)
    *   [3.6 User Interaction: Enabling and Configuring Chunking](#36-user-interaction-enabling-and-configuring-chunking)
4.  [Voice Cloning: Replicating Voices with Reference Audio](#4-voice-cloning-replicating-voices-with-reference-audio)
    *   [4.1 The Concept: Conditioning on a Voice Sample](#41-the-concept-conditioning-on-a-voice-sample)
    *   [4.2 Core Requirements for Cloning](#42-core-requirements-for-cloning)
    *   [4.3 How It Works: The Server-Side Cloning Pipeline](#43-how-it-works-the-server-side-cloning-pipeline)
    *   [4.4 User Interaction and API Usage](#44-user-interaction-and-api-usage)
        *   [Web UI Workflow](#web-ui-workflow)
        *   [API Workflow (`/tts` endpoint)](#api-workflow-tts-endpoint)
        *   [API Workflow (`/v1/audio/speech` endpoint)](#api-workflow-v1audiospeech-endpoint)
    *   [4.5 Tips for Quality Cloning](#45-tips-for-quality-cloning)
    *   [4.6 Limitations](#46-limitations)
5.  [Predefined Voices: Consistent Synthetic Voices](#5-predefined-voices-consistent-synthetic-voices)
    *   [5.1 Concept and Benefits](#51-concept-and-benefits)
    *   [5.2 Setup and Requirements](#52-setup-and-requirements)
    *   [5.3 Usage (UI & API)](#53-usage-ui--api)
6.  [Consistent Generation (Seeding)](#6-consistent-generation-seeding)
7.  [Audio Post-Processing](#7-audio-post-processing)
8.  [Visual Overview](#8-visual-overview)
    *   [8.1 Directory Structure](#81-directory-structure)
    *   [8.2 Component Diagram](#82-component-diagram)
9.  [System Prerequisites](#9-system-prerequisites)
10. [Installation and Setup](#10-installation-and-setup)
    *   [10.1 Cloning the Repository](#101-cloning-the-repository)
    *   [10.2 Setting up Python Virtual Environment](#102-setting-up-python-virtual-environment)
        *   [10.2.1 Windows Setup](#1021-windows-setup)
        *   [10.2.2 Linux Setup (Debian/Ubuntu Example)](#1022-linux-setup-debianubuntu-example)
    *   [10.3 Installing Dependencies](#103-installing-dependencies)
    *   [10.4 NVIDIA Driver and CUDA Setup (Required for GPU Acceleration)](#104-nvidia-driver-and-cuda-setup-required-for-gpu-acceleration)
        *   [10.4.1 Step 1: Check/Install NVIDIA Drivers](#1041-step-1-checkinstall-nvidia-drivers)
        *   [10.4.2 Step 2: Install PyTorch with CUDA Support](#1042-step-2-install-pytorch-with-cuda-support)
        *   [10.4.3 Step 3: Verify PyTorch CUDA Installation](#1043-step-3-verify-pytorch-cuda-installation)
11. [Configuration (`config.yaml`)](#11-configuration-configyaml)
    *   [11.1 Configuration File Overview (`config.yaml` vs `.env`)](#111-configuration-file-overview-configyaml-vs-env)
    *   [11.2 Configuration Parameters](#112-configuration-parameters)
    *   [11.3 Managing Configuration via UI](#113-managing-configuration-via-ui)
12. [Running the Server](#12-running-the-server)
13. [Usage](#13-usage)
    *   [13.1 Web User Interface (Web UI)](#131-web-user-interface-web-ui)
        *   [13.1.1 Main Generation Form](#1311-main-generation-form)
        *   [13.1.2 Text Splitting / Chunking Controls](#1312-text-splitting--chunking-controls)
        *   [13.1.3 Voice Mode Selection (Predefined, Clone, Random/Dialogue)](#1313-voice-mode-selection-predefined-clone-randomdialogue)
        *   [13.1.4 Presets](#1314-presets)
        *   [13.1.5 Generation Parameters (including Seeding)](#1315-generation-parameters-including-seeding)
        *   [13.1.6 Server Configuration (UI)](#1316-server-configuration-ui)
        *   [13.1.7 Generated Audio Player](#1317-generated-audio-player)
        *   [13.1.8 Theme Toggle](#1318-theme-toggle)
        *   [13.1.9 Session Persistence](#1319-session-persistence)
    *   [13.2 API Endpoints](#132-api-endpoints)
        *   [13.2.1 POST /v1/audio/speech (OpenAI Compatible)](#1321-post-v1audiospeech-openai-compatible)
        *   [13.2.2 POST /tts (Custom Parameters)](#1322-post-tts-custom-parameters)
        *   [13.2.3 Configuration & Helper Endpoints](#1323-configuration--helper-endpoints)
14. [Troubleshooting](#14-troubleshooting)
15. [Project Architecture](#15-project-architecture)
16. [License and Disclaimer](#16-license-and-disclaimer)

---

## 1. Overview

The Dia TTS Server provides a backend service and web interface for generating high-fidelity speech, including dialogue with multiple speakers and non-verbal sounds, using the Dia text-to-speech model family (originally from Nari Labs, with support for community conversions like SafeTensors).

This server is built using the FastAPI framework and offers both a RESTful API (including an OpenAI-compatible endpoint) and an interactive web UI powered by Jinja2, Tailwind CSS, and JavaScript. It supports **voice cloning via audio prompts and transcripts**, **consistent voice generation via predefined voices and seeding**, **automatic processing of large text inputs through intelligent, speaker-aware chunking**, and allows configuration of various generation parameters. Configuration is primarily managed via a `config.yaml` file.

**Key Features:**

*   **High-Quality TTS:** Leverages the Dia model for realistic speech synthesis.
*   **Dialogue Generation:** Supports `[S1]` and `[S2]` tags for multi-speaker dialogue.
*   **Non-Verbal Sounds:** Can generate sounds like `(laughs)`, `(sighs)`, etc.
*   **Voice Cloning:** Allows conditioning the output voice on a provided reference audio file and its transcript (local `.txt` recommended, Whisper fallback experimental). See Section 4.
*   **Predefined Voices:** Offers 43 ready-to-use synthetic voices for consistent output without cloning setup. See Section 5.
*   **Consistent Generation:** Use Predefined/Cloned voices and/or integer seeds for reproducible voice characteristics. See Section 6.
*   **Large Text Handling:** Automatically splits long text based on sentence structure and speaker tags, generates audio per chunk, and concatenates the results seamlessly. See Section 3.
*   **Flexible Model Loading:** Supports loading models from Hugging Face repositories (`.pth`, `.safetensors`). Defaults to BF16 SafeTensors.
*   **API Access:** Provides `/tts` (custom, with cloning/chunking/seeding) and `/v1/audio/speech` (OpenAI compatible, basic cloning/seeding) endpoints.
*   **Web Interface:** Offers an easy-to-use UI for text input, parameter adjustment (seed, chunking), preset loading, reference audio management (upload/selection), predefined voice selection, and audio playback. Includes session persistence via `config.yaml`.
*   **Configuration:** Server settings, model sources, paths, Whisper model, and default generation parameters are configurable via `config.yaml` (editable via UI or file). `.env` used for initial seeding/reset.
*   **GPU Acceleration:** Utilizes NVIDIA GPUs via CUDA, falling back to CPU. Optimized VRAM usage (~7GB typical).
*   **Performance:** Significant speed improvements (approaching 95% real-time on tested hardware).
*   **Audio Post-Processing:** Includes automatic silence trimming, internal silence fixing, and unvoiced segment removal. See Section 7.
*   **Terminal Progress:** Displays a progress bar (`tqdm`) when processing text chunks.

---

## 2. What's New (v1.4.0)

This version introduces significant improvements and new features compared to earlier versions (e.g., v1.0.0):

**🚀 New Features:**

*   **Large Text Processing (Chunking):**
    *   Implemented intelligent chunking based on sentence boundaries and speaker tags (`[S1]`/`[S2]`).
    *   Handles long text inputs automatically, overcoming previous limits.
    *   Configurable via UI toggle and chunk size slider. Detailed in Section 3.
*   **Predefined Voices:**
    *   Added support for 43 curated, ready-to-use synthetic voices (stored in `./voices`).
    *   Selectable via UI dropdown ("Predefined Voices" mode). Server automatically uses required transcripts.
    *   Provides reliable voice output without manual cloning setup and avoids potential licensing issues. Detailed in Section 5.
*   **Enhanced Voice Cloning:**
    *   Improved backend pipeline for robustness.
    *   Automatic reference audio processing (mono, resampling, truncation).
    *   Automatic transcript handling (local `.txt` recommended, Whisper fallback experimental). Backend handles transcript prepending. Detailed in Section 4.
*   **Whisper Integration:** Added `openai-whisper` for automatic transcript generation fallback during cloning. Configurable model (`model.whisper_model_name` in `config.yaml`).
*   **Generation Seed:** Added `seed` parameter to UI and API for influencing generation results. Helps maintain consistency when used with Predefined/Cloned voices. Detailed in Section 6.
*   **API Enhancements:**
    *   `/tts` endpoint now supports `transcript` (for explicit clone transcript), `split_text`, `chunk_size`, and `seed`.
    *   `/v1/audio/speech` endpoint now supports `seed`.
*   **Terminal Progress:** Generation of long text (using chunking) now displays a `tqdm` progress bar.
*   **UI Configuration Management:** Added UI section to view/edit `config.yaml` settings and save generation defaults.
*   **Configuration System:** Migrated to `config.yaml` for primary runtime configuration, managed via `config.py` (`YamlConfigManager`). `.env` is now used mainly for initial seeding or resetting defaults.

**🔧 Fixes & Enhancements:**

*   **VRAM Usage Fixed & Optimized:** Resolved memory leaks and significantly reduced VRAM usage (approx. 14GB+ down to ~7GB) through code optimizations, fixing memory leaks, and BF16 default.
*   **Performance:** Significant speed improvements reported (approaching 95% real-time on tested hardware: AMD Ryzen 9 9950X3D + NVIDIA RTX 3090).
*   **Audio Post-Processing:** Automatically applies silence trimming, internal silence fixing, and unvoiced segment removal (using Parselmouth) to improve audio quality. Detailed in Section 7.
*   **UI State Persistence:** Web UI now saves/restores settings (text, mode, files, parameters) in `config.yaml`.
*   **UI Improvements:** Better loading indicators (shows chunk processing), refined chunking controls, seed input, theme toggle, dynamic preset loading from `ui/presets.yaml`, warning modals.
*   **Cloning Workflow:** Backend now handles transcript prepending automatically. UI workflow simplified.
*   **Dependency Management:** Added `tqdm`, `PyYAML`, `openai-whisper`, `parselmouth`.
*   **Code Refactoring:** Aligned internal engine code with refactored `dia` library structure. Updated `config.py` to use `YamlConfigManager`.

---

## 3. Large Text Processing & Chunking: Generating Long Audio Naturally

Generating high-quality audio for lengthy texts, like articles, book chapters, or extended dialogues, presents unique challenges for Text-to-Speech (TTS) systems. This server incorporates an advanced chunking mechanism specifically designed to address these challenges, ensuring both technical feasibility and natural-sounding results.

### 3.1 The Challenge: Why Long Text is Hard for TTS

Simply feeding a very long piece of text directly into a TTS model often leads to problems:

*   **Technical Limits:** TTS models, including Dia, have inherent limitations on the amount of text they can process in one go. Exceeding this limit typically causes errors or incomplete audio generation.
*   **Memory Constraints:** Processing large amounts of text simultaneously demands significant computational resources, particularly VRAM on GPUs. This can lead to "Out of Memory" errors or slow down generation considerably.
*   **Loss of Natural Flow (Prosody):** If text is split arbitrarily (e.g., every N characters) without considering its linguistic structure, the resulting audio segments, when concatenated, can sound disjointed. Pauses might occur mid-sentence, intonation contours can be broken, and the overall rhythm and flow (prosody) of speech are disrupted, leading to robotic or unnatural output.
*   **Speaker Inconsistency in Dialogue:** For texts with multiple speakers indicated by tags like `[S1]` and `[S2]`, naive splitting can easily place text from different speakers into the same chunk. This confuses the TTS model, which expects a single voice per input, resulting in inconsistent or incorrect voice generation for that segment.

### 3.2 Our Solution: Intelligent, Context-Aware Chunking

To overcome these hurdles, the Dia TTS Server implements a sophisticated chunking strategy that prioritizes linguistic and structural coherence over simple length constraints. It operates on two fundamental principles:

1.  **Respecting Sentence Boundaries:** The system recognizes that sentences are the basic building blocks of coherent speech. It uses advanced logic (`split_into_sentences` in `utils.py`) to identify sentence endings (`.`, `!`, `?`) accurately, carefully handling potential ambiguities like abbreviations ("Dr. Smith") or decimal points ("version 2.5"). Sentences are treated as indivisible units during the chunking process.
2.  **Enforcing Speaker Homogeneity:** In dialogue scenarios, maintaining distinct voices is paramount. The chunking mechanism strictly enforces that **each generated chunk contains text attributable to only *one* speaker tag (`[S1]` or `[S2]`)**. A change in speaker tag automatically triggers a chunk break, regardless of length.

A configurable character limit (`chunk_size`) acts as a secondary mechanism to prevent individual chunks (within a single speaker's turn) from becoming too long for the TTS engine or consuming excessive memory.

### 3.3 The Chunking Process: A Step-by-Step Look

When chunking is enabled (via the "Split text into chunks" UI option or `split_text=True` API parameter), the server executes a multi-step process orchestrated by the `chunk_text_by_sentences` function in `utils.py`:

#### Step 1: Understanding the Structure - Tagging and Sentence Splitting

*   **Input:** The raw text provided by the user.
*   **Action (`_preprocess_and_tag_sentences`):**
    *   The text is first scanned to identify all speaker tags (`[S1]`, `[S2]`, etc.) using regular expressions.
    *   The system determines the "active" speaker for different parts of the text. It defaults to `[S1]` if no tag is present at the beginning and remembers the last seen tag for subsequent text until a new tag appears.
    *   The text *content* between these tags (or from the start/to the end) is extracted, effectively removing the tags from the text itself for now.
    *   This tag-free content is then passed to the `split_into_sentences` function. This function intelligently breaks the content into individual sentences, carefully avoiding splits after abbreviations or within numbers. It also handles splitting based on bullet points.
    *   Finally, each identified sentence is paired with the speaker tag that was active for its original segment.
*   **Output:** A structured list where each item represents a sentence ready for processing, containing both the speaker tag and the clean sentence text. For example: `[ ('[S1]', 'This is the first sentence.'), ('[S1]', 'This is the second.'), ('[S2]', 'Now speaker two talks.') ]`

#### Step 2: Building the Chunks - Aggregation with Rules

*   **Input:** The list of tagged sentences from Step 1.
*   **Action (`chunk_text_by_sentences` main loop):**
    *   The system iterates through the tagged sentences, attempting to group consecutive sentences into chunks.
    *   It maintains a `current_chunk_text` and the `current_chunk_tag`.
    *   **Rule Check 1 (Speaker Change):** Before adding the *next* sentence, its tag is checked. If `next_sentence_tag` is different from `current_chunk_tag`, the system knows the speaker has changed. The `current_chunk` is considered complete and is finalized (prepared for output in Step 3). A new chunk is then started using the `next_sentence` and its tag. This is the highest priority rule.
    *   **Rule Check 2 (Length Limit):** If the `next_sentence_tag` *matches* the `current_chunk_tag`, the system checks if adding the `next_sentence` (plus a space for joining) would make the `current_chunk_text` exceed the `chunk_size` limit. If it *would* exceed the limit, the `current_chunk` is finalized, and a new chunk is started with the `next_sentence` (using the same speaker tag).
    *   **Aggregation:** If the speaker tags match *and* the combined length is within the `chunk_size` limit, the `next_sentence` text is appended (with a preceding space) to the `current_chunk_text`.
*   **Output:** A list of text segments, where each segment contains one or more sentences from a single speaker and respects the length constraint where possible.

#### Step 3: Final Touches - Formatting the Output

*   **Input:** The finalized text segments from Step 2.
*   **Action:** As each chunk is finalized in Step 2, the system takes the associated `current_chunk_tag` (e.g., `[S1]`) and prepends it, followed by a space, to the aggregated `current_chunk_text`.
*   **Output:** The final list of strings ready to be sent to the TTS engine, one by one. Each string starts with the correct speaker tag and contains a linguistically coherent segment of the original text. Example: `[ "[S1] This is the first sentence. This is the second.", "[S2] Now speaker two talks." ]`

#### Example Walkthrough

Consider this input text with `chunk_size = 100`:

```
[S1] This is the first sentence, it's quite short. This is the second sentence, making the first speaker's turn a bit longer now. [S2] Speaker two starts here. This is speaker two's second sentence.
```

1.  **Step 1 (Tag & Split):**
    *   Output List: `[ ('[S1]', "This is the first sentence, it's quite short."), ('[S1]', "This is the second sentence, making the first speaker's turn a bit longer now."), ('[S2]', "Speaker two starts here."), ('[S2]', "This is speaker two's second sentence.") ]`

2.  **Step 2 (Aggregate):**
    *   Start Chunk 1: `tag='[S1]', text="This is the first sentence, it's quite short."` (Len 46).
    *   Consider next `('[S1]', "Sentence A2")`: Tag matches. Potential length = 46 + 1 + 80 = 127. Exceeds `chunk_size` (100).
    *   Finalize Chunk 1: Text is "Sentence A1".
    *   Start Chunk 2: `tag='[S1]', text="Sentence A2"` (Len 80).
    *   Consider next `('[S2]', "Sentence B1")`: Tag `[S2]` differs from `[S1]`. Speaker change!
    *   Finalize Chunk 2: Text is "Sentence A2".
    *   Start Chunk 3: `tag='[S2]', text="Sentence B1"` (Len 25).
    *   Consider next `('[S2]', "Sentence B2")`: Tag matches. Potential length = 25 + 1 + 37 = 63. <= `chunk_size` (100).
    *   Aggregate: `text` becomes "Sentence B1 Sentence B2".
    *   End of input. Finalize Chunk 3: Text is "Sentence B1 Sentence B2".

3.  **Step 3 (Format):**
    *   Chunk 1: `"[S1] This is the first sentence, it's quite short."`
    *   Chunk 2: `"[S1] This is the second sentence, making the first speaker's turn a bit longer now."`
    *   Chunk 3: `"[S2] Speaker two starts here. This is speaker two's second sentence."`

*   **Final Output Chunks:**
    1.  `"[S1] This is the first sentence, it's quite short."`
    2.  `"[S1] This is the second sentence, making the first speaker's turn a bit longer now."`
    3.  `"[S2] Speaker two starts here. This is speaker two's second sentence."`

Notice how the speaker change forced a break between chunks 2 and 3, and the `chunk_size` forced a break between chunks 1 and 2, even though they had the same speaker.

### 3.4 The `chunk_size` Parameter: Controlling Segment Length

While speaker tags and sentence boundaries provide the primary structure, the `chunk_size` parameter offers essential control over the maximum length of segments, particularly during longer passages spoken by a single speaker.

*   **Secondary Role:** It acts *after* the speaker tag rule. If the speaker changes, a break occurs regardless of `chunk_size`.
*   **Length Governor:** Its main purpose is to prevent chunks containing multiple sentences from the *same speaker* from exceeding a practical length for the TTS engine. It acts as an upper bound on aggregation *within* a speaker's turn.
*   **Impact:**
    *   A **lower `chunk_size`** (e.g., 100) will cause more frequent breaks during long monologues, resulting in more, shorter audio segments. This can be useful if the TTS engine has stricter limits or if smaller audio files are preferred. Recommended range: ~100-300.
    *   A **higher `chunk_size`** (e.g., 350) allows more sentences from the same speaker to be grouped together, leading to fewer, longer audio segments, potentially offering slightly better contextual flow within that speaker's turn, as long as it stays within the engine's limits.
*   **Oversized Sentence Handling:** It's important to remember that `chunk_size` does *not* force splits *within* a sentence. If a single sentence naturally exceeds the `chunk_size` (e.g., a 500-character sentence with `chunk_size` set to 400), that sentence will still form its own chunk. The system prioritizes sentence integrity over the length limit in this specific scenario, and a warning will be logged.

### 3.5 Why This Matters: Benefits of Smart Chunking

This intelligent chunking approach provides several key advantages:

*   **Natural Sounding Audio:** By respecting sentence structure and speaker turns, the breaks between concatenated audio segments occur at more logical points, significantly improving the naturalness and flow of the final long-form audio.
*   **Dialogue Consistency:** Guarantees that the correct voice is used for each speaker throughout the dialogue by ensuring each chunk is homogenous.
*   **Error Prevention:** Reliably keeps individual text segments within the processing limits of the TTS engine, preventing errors and ensuring complete generation.
*   **Improved Resource Management:** Processing moderately sized chunks can be more efficient in terms of memory usage compared to attempting very large segments.
*   **Robustness:** Handles various text structures, including monologues, dialogues, and mixed inputs, gracefully.

### 3.6 User Interaction: Enabling and Configuring Chunking

Users can easily control this feature:

*   **Web UI:**
    *   The **"Split text into chunks" checkbox** (enabled by default) toggles the entire chunking mechanism on or off. If disabled, the server will attempt to process the entire text at once (which may fail if it exceeds limits).
    *   The **"Chunk Size" slider** (visible when splitting is enabled) allows adjusting the character limit used for same-speaker aggregation (default 120, min 50, max 400).
*   **API (`/tts` endpoint):**
    *   The `split_text: true` (default) or `split_text: false` parameter controls whether chunking is performed.
    *   The `chunk_size: <integer>` parameter (default 120) sets the desired length limit when `split_text` is true.

This combination of automatic intelligence and user configuration provides a powerful and flexible solution for generating high-quality, long-form audio.

---

## 4. Voice Cloning: Replicating Voices with Reference Audio

Beyond generating speech with its default or predefined voices, the Dia TTS Server supports **voice cloning**, also known as zero-shot Text-to-Speech. This powerful feature allows you to generate speech in a voice that mimics a provided audio sample, enabling personalized voice outputs without needing to retrain the model.

### 4.1 The Concept: Conditioning on a Voice Sample

Voice cloning works by providing the TTS model with two key pieces of information about the desired voice *in addition* to the text you want it to speak:

1.  **Reference Audio:** A short audio clip (typically 5-20 seconds) containing speech from the target voice. This sample gives the model acoustic information about the voice's characteristics (pitch, tone, timbre, rhythm).
2.  **Reference Transcript:** The exact text spoken in the reference audio clip. This helps the model associate the acoustic features with the corresponding phonetic content, improving the accuracy of the voice replication.

The Dia model uses these inputs during generation to "condition" its output, aiming to produce the *target text* spoken in a voice similar to the *reference audio*.

### 4.2 Core Requirements for Cloning

To successfully use the voice cloning feature, the user must provide:

1.  **A Reference Audio File:**
    *   Format: `.wav` or `.mp3`.
    *   Location: Must be placed within the server's configured `paths.reference_audio` directory (default: `./reference_audio/`). Files can be added manually or uploaded via the Web UI's "Import" button in the Clone section.
    *   Quality: Clear speech with minimal background noise yields the best results.
    *   Duration: Ideally between 5 and 20 seconds. The server will automatically truncate longer files.
2.  **The Target Text:** The new text you want the cloned voice to speak (entered in the main text box or API `text` field).
3.  **The Reference Transcript:** The *exact* word-for-word transcription of what is spoken in the reference audio file. This is crucial for the model to learn the voice-text mapping effectively.
    *   **Recommended Method:** Create a text file (`.txt`) with the **exact same base name** as the reference audio file (e.g., `my_voice.wav` needs `my_voice.txt`) and place it in the `paths.reference_audio` directory.
    *   **Transcript Format:** The `.txt` file **must** include speaker tags:
        *   For single speaker audio: `[S1] Transcript text here.`
        *   For two speakers in reference: `[S1] First speaker text. [S2] Second speaker text.`
    *   **Automatic Fallback (Experimental):** If the `.txt` file is missing, the server will *attempt* to generate the transcript using Whisper (see 4.3). This is less reliable than providing the `.txt` file.
    *   **API Override:** The `/tts` API endpoint allows providing the transcript directly via the `transcript` field, overriding file/Whisper lookup.

### 4.3 How It Works: The Server-Side Cloning Pipeline

When a request is made with `voice_mode="clone"` (either via API or the Web UI), the server executes a specific preparation pipeline within the `engine.py` module:

1.  **Reference File Validation (`_find_reference_file`):**
    *   The server takes the provided `clone_reference_filename`.
    *   It robustly searches the `paths.reference_audio` directory, ignoring case and potential user-provided extensions (`.wav`/`.mp3` or none). It handles leading/trailing whitespace and removes any path components, focusing only on the base filename.
    *   If a unique, valid `.wav` or `.mp3` file matching the base name is found, its actual filename (with correct casing) and full path are used.
    *   If no match or multiple ambiguous matches are found, the process fails, and an error is returned.

2.  **Audio Loading and Processing:**
    *   The validated reference audio file is loaded into memory using `torchaudio.load`.
    *   **Resampling:** If the audio's sample rate doesn't match the model's expected rate (`EXPECTED_SAMPLE_RATE`, typically 44100 Hz), it's automatically resampled.
    *   **Mono Conversion:** If the audio is stereo, it's converted to mono by averaging the channels. A warning is logged.
    *   **Truncation:** If the audio duration exceeds a predefined limit (`max_ref_duration_sec`, currently ~20 seconds), the audio is truncated from the beginning. A warning is logged. This ensures the audio prompt isn't excessively long. The resulting processed audio data (as a NumPy array) is stored temporarily in RAM for potential Whisper use.

3.  **Transcript Acquisition:** The server attempts to get the reference transcript using the following priority order:
    *   **Priority 1: Explicit Transcript (API Only):** If the `/tts` API request includes a non-null `transcript` field in its JSON body, the server uses this text directly. It checks if the provided transcript starts with `[S1]` or `[S2]`; if not, it automatically prepends `[S1] ` for compatibility.
    *   **Priority 2: Local `.txt` File:** If no explicit transcript was provided, the server constructs the expected transcript filename (e.g., `my_voice.txt` for `my_voice.wav`) and looks for it in the `paths.reference_audio` directory. If found, its content is read and used directly. **This is the recommended method.** The `.txt` file must be correctly formatted with speaker tags.
    *   **Priority 3: Whisper Generation (Experimental Fallback):** If no explicit transcript was provided *and* the local `.txt` file was not found, the server attempts automatic transcription. It uses the `openai-whisper` library (`_generate_transcript_with_whisper` in `utils.py`) to transcribe the *processed* (mono, truncated) audio data held in RAM.
        *   The specific Whisper model used is determined by the `model.whisper_model_name` setting in `config.yaml` (default: `small.en`). The model is loaded from the `paths.model_cache`.
        *   If transcription is successful, the server automatically prepends `[S1] ` to the generated text.
        *   The generated (and tagged) transcript is then saved back to the corresponding `.txt` file in `paths.reference_audio` for future use, overwriting any previous placeholder if necessary.
    *   **Error Handling:** If none of the above methods yield a transcript, the process fails, and an error is returned.

4.  **Input Combination & Generation:**
    *   The successfully obtained reference transcript (from API, file, or Whisper, now guaranteed to have a speaker tag) is automatically prepended to the user's *target text* (the text they entered in the UI or API `text` field), separated by a space.
    *   Example: `"[S1] This is the reference transcript text. [S1] This is the target text the user wants generated."`
    *   This combined text becomes the final input for the model's text encoder.
    *   The **path** to the validated reference audio file is passed to the `dia_model.generate` function.
    *   `dia_model.generate` internally loads the audio, encodes it into an `audio_prompt` tensor, and uses both the combined text and the audio prompt to condition the generation process.
    *   If chunking is enabled, the combined text is split, and the audio prompt conditioning is applied primarily to the first chunk, with subsequent chunks maintaining context.

### 4.4 User Interaction and API Usage

#### Web UI Workflow

1.  Select the "Voice Clone" radio button.
2.  Use the "Reference Audio File" dropdown to select a file previously placed in or uploaded to the `reference_audio` directory. Use the "Import" button to upload new `.wav` or `.mp3` files.
3.  **Ensure the corresponding `.txt` transcript file exists** in the `reference_audio` directory (recommended) or rely on the experimental Whisper fallback if the `.txt` is missing. The transcript file must be formatted correctly (e.g., `[S1] text...`).
4.  Enter **only the target text** you want the cloned voice to speak into the main "Text to speak" area. The backend handles prepending the transcript automatically.
5.  Adjust generation parameters (Seed, Chunking, etc.) as needed.
6.  Click "Generate Speech".

#### API Workflow (`/tts` endpoint)

This endpoint offers the most control over cloning.

1.  Set `"voice_mode": "clone"`.
2.  Provide `"clone_reference_filename": "your_reference_file.wav"` (must exist on the server in `paths.reference_audio`).
3.  **Choose Transcript Method:**
    *   **Option 1 (Recommended for Remote/Accuracy):** Provide the `"transcript": "Exact transcript text here..."` field in the JSON body. The server will use this directly. Ensure it includes speaker tags (`[S1]`/`[S2]`).
    *   **Option 2 (Server-Side Lookup/Generation):** Omit the `"transcript"` field. The server will first look for `your_reference_file.txt` in `paths.reference_audio`. If not found, it will attempt Whisper generation (experimental).
4.  Provide the target `"text": "Text to be generated..."`. The server handles prepending the transcript internally before generation.
5.  Set other parameters (`seed`, `split_text`, `chunk_size`, etc.) as needed.

#### API Workflow (`/v1/audio/speech` endpoint)

This endpoint has limited cloning capabilities.

1.  Set the `"voice"` parameter to the filename of the reference audio, e.g., `"voice": "your_reference_file.wav"`. The file must exist in `paths.reference_audio`.
2.  **You must manually prepend the exact transcript** to the main text in the `"input"` field. Example: `"input": "[S1] Reference transcript text here. [S1] Target text to be generated..."`.
3.  This endpoint does **not** support the explicit `transcript` field and does **not** perform automatic `.txt` lookup or Whisper generation. Transcript prepending is mandatory in the `input`.
4.  Set `seed` if desired. Chunking uses default settings.

### 4.5 Tips for Quality Cloning

*   **Reference Audio Quality:** Use clear recordings with minimal noise, reverb, or music. A single speaker is ideal.
*   **Transcript Accuracy:** Ensure the transcript in the `.txt` file (or provided via API) is *exact*. Even small deviations can impact quality. Use the correct `[S1]`/`[S2]` format.
*   **Reference Duration:** 10-20 seconds is often a good balance. Too short might not capture enough voice character; too long is truncated and might contain unwanted variations.
*   **Parameter Tuning:** Experiment with `temperature` and `cfg_scale`. Sometimes slightly lower values can produce a clone closer to the reference, while higher values might sound more expressive but less similar.
*   **Seed:** Using a fixed seed can help if you find a particular seed yields a better clone for a specific reference voice.

### 4.6 Limitations

Zero-shot voice cloning is a challenging task. While Dia can produce impressive results, the quality and similarity depend heavily on the reference audio and transcript quality. It may not perfectly capture every subtle nuance, accent, or emotional tone of the original speaker. Expect good similarity rather than a perfect replication. Whisper fallback for transcripts is experimental and may reduce quality compared to providing an accurate `.txt` file.

---

## 5. Predefined Voices: Consistent Synthetic Voices

To simplify generating speech with consistent, high-quality voices without the complexities of voice cloning, the server offers a "Predefined Voices" mode.

### 5.1 Concept and Benefits

*   **Curated Voices:** This mode utilizes a set of ready-to-use voice samples stored on the server. The project includes 43 such voices by default.
*   **Synthetic Origin:** These voices are typically synthetic creations, meaning they do not directly mimic specific real individuals. This avoids potential ethical concerns and licensing issues associated with cloning real voices without permission.
*   **Ease of Use:** Users simply select a named voice from the UI dropdown (e.g., "Michael-Emily", "Sarah"). The server handles finding the corresponding audio sample and its required transcript automatically.
*   **Consistency:** Provides a reliable way to achieve consistent voice output for projects requiring a specific vocal characteristic, especially when combined with a fixed generation seed (see Section 6).

### 5.2 Setup and Requirements

*   **Directory:** Predefined voice files must be placed in the directory specified by `paths.voices` in `config.yaml` (default: `./voices/`).
*   **Audio Files:** Each voice requires a `.wav` audio sample (typically 5-20 seconds).
*   **Transcript Files:** **Crucially, each `.wav` file MUST have a corresponding `.txt` file with the exact same base name** (e.g., `Michael_Emily.wav` requires `Michael_Emily.txt`).
    *   The `.txt` file must contain the exact transcript of the audio, formatted with speaker tags: `[S1] Speaker one text.` or `[S1] Speaker one text. [S2] Speaker two text.`
    *   The server relies on these `.txt` files; it does **not** use Whisper for predefined voices.
*   **Naming:** Files can be named descriptively (e.g., `Narrator_Deep.wav`, `Character_Female_Energetic.wav`, `Speaker1_Speaker2_Dialogue.wav`). Underscores in filenames are converted to hyphens for display names in the UI (e.g., `Michael_Emily.wav` appears as "Michael-Emily").

### 5.3 Usage (UI & API)

*   **Web UI:**
    1.  Select the "Predefined Voices" radio button.
    2.  Choose the desired voice from the "Select Predefined Voice" dropdown.
    3.  Enter the target text to be spoken in the main text box.
    4.  Adjust generation parameters (Seed, Chunking, etc.) as needed.
    5.  Click "Generate Speech".
*   **API (`/tts` endpoint):**
    1.  Set `"voice_mode": "predefined"`.
    2.  Provide the exact filename of the desired voice file (e.g., `"Michael_Emily.wav"`) in the `"clone_reference_filename"` field.
    3.  Provide the target `"text"`.
    4.  Set other parameters (`seed`, `split_text`, etc.) as needed.
*   **API (`/v1/audio/speech` endpoint):**
    1.  Set the `"voice"` parameter to the exact filename of the desired predefined voice file (e.g., `"voice": "Michael_Emily.wav"`).
    2.  Provide the target `"input"`.
    3.  Set `seed` if desired.

---

## 6. Consistent Generation (Seeding)

To achieve reproducible and consistent voice output, especially across multiple generations or when using text chunking, the server utilizes a generation **seed**.

*   **Concept:** The seed is an integer value that initializes the random number generator used during the TTS model's sampling process. Using the same seed with the same input text and parameters (including the same voice mode selection - Predefined or Clone with the same reference) will generally produce very similar, if not identical, audio output.
*   **Randomness:** Using a seed value of `-1` tells the server to use a random seed for each generation, resulting in variations in voice delivery and characteristics (unless using Predefined/Cloned voices which inherently provide some consistency).
*   **Consistency:** Providing a specific integer (e.g., `42`, `12345`, `901`) ensures that the random aspects of the generation process are fixed, leading to consistent output. This is crucial when:
    *   Generating long audio using the **chunking** feature to maintain the same voice across all chunks.
    *   Regenerating audio for a specific script multiple times.
    *   Trying to find and reuse a specific voice characteristic discovered during experimentation.
*   **Usage:**
    *   **Web UI:** Enter the desired integer seed in the "Generation Seed" input field within the "Generation Parameters" section. Use `-1` for random.
    *   **API (`/tts` & `/v1/audio/speech`):** Include the `"seed": <integer>` parameter in your JSON request body. Use `-1` for random.

**Note:** While seeding significantly increases consistency, minor variations might still occur due to floating-point arithmetic differences across hardware or software versions. However, for practical purposes, using a fixed seed with Predefined or Cloned voices provides reliable voice consistency.

---

## 7. Audio Post-Processing

To improve the quality and usability of the generated audio, the server automatically applies several post-processing steps after the core TTS generation is complete. These steps are always active and not currently user-configurable.

1.  **Leading/Trailing Silence Trimming:** Removes excessive silence from the beginning and end of the audio clip. Helps ensure the audio starts and ends promptly.
2.  **Internal Silence Reduction:** Shortens long pauses *within* the speech to a more natural maximum duration (e.g., reduces a 3-second pause to 0.5 seconds). Prevents unnaturally long gaps.
3.  **Unvoiced Segment Removal (Parselmouth):** Attempts to identify and remove long segments of audio that contain no voiced speech (e.g., long breaths, background noise without speech, some audio artifacts). This relies on the `parselmouth` library (Praat integration) and helps clean up the audio.

These steps are applied sequentially to the final concatenated audio (if chunking was used) or the single generated audio clip. Logs indicating the changes made (e.g., percentage change in length) may appear in the server console.

---

## 8. Visual Overview

### 8.1 Directory Structure

```dia-tts-server/
│
├── .env                  # Initial configuration seeding (optional, read only once if config.yaml missing)
├── config.yaml           # PRIMARY configuration file (created/managed by server)
├── config.py             # Default config values & YamlConfigManager class
├── engine.py             # Core model loading, generation logic, chunking, cloning prep
├── models.py             # Pydantic models for API requests
├── requirements.txt      # Python dependencies (incl. tqdm, whisper, yaml, parselmouth)
├── server.py             # Main FastAPI application, API endpoints, UI routes
├── utils.py              # Utility functions (audio encoding, saving, chunking, whisper, sanitize, post-proc)
│
├── dia/                  # Core Dia model implementation package (Refactored Version)
│   ├── __init__.py
│   ├── audio.py          # Low-level audio processing (delay patterns)
│   ├── config.py         # Pydantic models for Dia model architecture config
│   ├── layers.py         # Custom PyTorch layers (Attention, MLP, RoPE)
│   ├── model.py          # Dia model class wrapper (loading, core step logic)
│   └── state.py          # Inference state management classes (KVCache, *InferenceState)
│
├── ui/                   # Web User Interface files
│   ├── index.html        # Main HTML template (Jinja2)
│   ├── presets.yaml      # Predefined UI examples (loaded dynamically)
│   └── script.js         # Frontend JavaScript logic
│
├── model_cache/          # Default directory for downloaded models (Dia & Whisper)
├── outputs/              # Default directory for saved audio output
├── reference_audio/      # Default directory for voice cloning reference files (.wav, .mp3, .txt)
└── voices/               # Default directory for predefined voice files (.wav, .txt)
```

### 8.2 Component Diagram

```
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────────────┐      ┌───────────────────┐
│ User (Web UI /    │────→ │ FastAPI Server    │────→ │ TTS Engine (Looping)      │────→ │ Dia Model Wrapper │
│ API Client)       │      │ (server.py)       │      │ (engine.py)               │      │ (dia/model.py)    │
└───────────────────┘      └─────────┬─────────┘      └──────┬─────────┬──────────┘      └─────────┬─────────┘
                                     │                      │ Calls   │ Calls                      │
                                     │ Uses                 │         │                            │ Uses
                                     ▼                      ▼         ▼                            ▼
                           ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
                           │ Configuration     │ ←─ │ config.yaml     │  │ Utilities         │  │ Dia Model Layers  │
                           │ (config.py)       │  └───────────────────┘  │ (utils.py)        │  │ (dia/layers.py)   │
                           └───────────────────┘                         │ - Chunking Logic  │  └─────────┬─────────┘
                                     ▲                                   │ - Whisper Transcr.│            │ Uses
                                     │ Uses                              │ - Audio Post-Proc │            │
                                     │                                   │ - File Handling   │            │ Uses
                                     │ Uses                              └──────┬────────────┘            │
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐      │ Uses                    ▼
│ Web UI Files      │ ←─── │ Jinja2 Templates  │      │ Whisper Model     │←─────┘          ┌───────────────────┐
│ (ui/)             │      └───────────────────┘      │ (openai-whisper)  │                 │ PyTorch / CUDA    │
└───────────────────┘               ▲                 └───────────────────┘                 └─────────┬─────────┘
                                    │ Renders                                                         │ Uses
                                    │                                                                 │
                                    └─────────────────────────────────────────────────────────────────┴──┐
                                                                                                         │ Uses
                                                                                                         ▼
                                                                                                 ┌───────────────────┐
                                                                                                 │ DAC Model         │
                                                                                                 │ (descript-audio..)│
                                                                                                 └───────────────────┘
```

**Diagram Legend:**

*   Boxes represent major components or file groups.
*   Arrows (`→`) indicate primary data flow or control flow.
*   Lines with "Uses" / "Calls" indicate dependencies or function calls.
*   Note the central role of `config.yaml` and the interaction of Utilities with Whisper and Post-Processing.

---

## 9. System Prerequisites

Before installing and running the Dia TTS Server, ensure your system meets the following requirements:

*   **Operating System:**
    *   Windows 10/11 (64-bit)
    *   Linux (Debian/Ubuntu recommended)
*   **Python:** Python 3.10 or later.
*   **Version Control:** Git.
*   **Internet Connection:** Required for downloads (Dia models, Whisper models, dependencies).
*   **(Optional but Recommended for Performance):**
    *   **NVIDIA GPU:** CUDA-compatible (Maxwell or newer). BF16 Dia model needs ~7GB VRAM (optimized). More VRAM is beneficial.
    *   **NVIDIA Drivers:** Latest version.
    *   **CUDA Toolkit:** Version compatible with PyTorch build (e.g., 11.8, 12.1).
*   **(Linux System Libraries):**
    *   `libsndfile1`: Required by `soundfile`. Install via package manager (e.g., `sudo apt install libsndfile1`).
    *   `ffmpeg`: Required by `openai-whisper` for robust audio loading. Install via package manager (e.g., `sudo apt install ffmpeg`).

---

## 10. Installation and Setup

Follow these steps carefully to get the server running.

### 10.1 Cloning the Repository

```bash
git clone https://github.com/devnen/dia-tts-server.git # Replace if needed
cd dia-tts-server
```

### 10.2 Setting up Python Virtual Environment

Using a virtual environment is strongly recommended.

#### 10.2.1 Windows Setup
```powershell
# In the dia-tts-server directory
python -m venv venv
.\venv\Scripts\activate
# Your prompt should now start with (venv)
```

#### 10.2.2 Linux Setup (Debian/Ubuntu Example)
```bash
# Ensure prerequisites are installed
sudo apt update && sudo apt install python3 python3-venv python3-pip libsndfile1 ffmpeg -y

# In the dia-tts-server directory
python3 -m venv venv
source venv/bin/activate
# Your prompt should now start with (venv)
```

### 10.3 Installing Dependencies

With the virtual environment activated (`(venv)` prefix visible):

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install project requirements (includes tqdm, yaml, parselmouth etc.)
pip install -r requirements.txt
```
⭐ **Note:** This installation includes large libraries like PyTorch. The download and installation process may take **several minutes** depending on your internet speed and system performance.

⭐ **Important:** This installs the *CPU-only* version of PyTorch by default. If you have an NVIDIA GPU, proceed to [Section 10.4](#104-nvidia-driver-and-cuda-setup-required-for-gpu-acceleration) **before** running the server for GPU acceleration.

### 10.4 NVIDIA Driver and CUDA Setup (Required for GPU Acceleration)

Skip this step if you only have a CPU.

#### 10.4.1 Step 1: Check/Install NVIDIA Drivers
Run `nvidia-smi`. Install/update if needed from NVIDIA website and reboot. Note the CUDA version supported by the driver.

#### 10.4.2 Step 2: Install PyTorch with CUDA Support
Go to [PyTorch website](https://pytorch.org/get-started/locally/), select Stable, OS, Pip, Python, and a CUDA version **<=** your driver's supported version. Copy the command.
In activated venv:
```bash
# Uninstall the CPU version first!
pip uninstall torch torchvision torchaudio -y

# Paste and run the command copied from the PyTorch website
# Example (replace with your actual command):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 10.4.3 Step 3: Verify PyTorch CUDA Installation
In activated venv, run `python` and execute the following single line:
```python
import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"Device name: {torch.cuda.get_device_name(0)}") if torch.cuda.is_available() else None; exit()
```
Ensure `CUDA available:` is `True`.

---

## 11. Configuration (`config.yaml`)

The server's primary configuration is managed through the `config.yaml` file located in the project root.

### 11.1 Configuration File Overview (`config.yaml` vs `.env`)

*   **`config.yaml` (Primary):**
    *   This file stores the active configuration used by the server at runtime.
    *   It includes server settings, model paths, generation defaults, and UI state persistence.
    *   It is **created automatically** by the server on its first run if it doesn't exist, using defaults from `config.py` potentially seeded by `.env`.
    *   You can **edit this file directly** to make persistent changes.
    *   Changes made via the **Web UI's configuration sections are saved directly to this file.**
    *   **Restart required** for changes in `server`, `model`, or `paths` sections.
*   **`.env` (Initial Seeding / Reset Only):**
    *   This file is **only read** under two specific conditions:
        1.  When `config.yaml` is missing (e.g., first run), its values override the hardcoded defaults to create the initial `config.yaml`.
        2.  When the "Reset All Settings" button is clicked in the UI, `config.yaml` is regenerated using hardcoded defaults overridden by `.env` values.
    *   It is **not read during normal server operation** once `config.yaml` exists.
    *   Useful for setting up initial deployment configurations or defining reset points.

### 11.2 Configuration Parameters

The `config.yaml` file uses a nested structure. Below are the main sections and parameters (defaults shown might be from `config.py` if not overridden):

```yaml
# config.yaml (Example Structure - Values may differ)
server:
  host: 0.0.0.0         # Server listen address
  port: 8003            # Server listen port
model:
  repo_id: ttj/dia-1.6b-safetensors # HF Repo for Dia model
  config_filename: config.json      # Model config filename in repo
  weights_filename: dia-v0_1_bf16.safetensors # Model weights filename in repo
  whisper_model_name: small.en      # Whisper model for cloning fallback ('tiny.en', 'base.en', etc.)
paths:
  model_cache: ./model_cache        # Cache for downloaded models (Dia & Whisper)
  reference_audio: ./reference_audio # Folder for voice cloning files (.wav/.mp3, .txt)
  output: ./outputs                 # Folder for generated audio saved from UI
  voices: ./voices                  # Folder for predefined voice files (.wav, .txt)
generation_defaults:                # Default values loaded by UI sliders/inputs
  speed_factor: 1.0
  cfg_scale: 3.0
  temperature: 1.3
  top_p: 0.95
  cfg_filter_top_k: 35
  seed: 42                          # Default seed (-1 for random)
  split_text: true                  # Default chunking state
  chunk_size: 120                   # Default chunk size
ui_state:                           # Saved state from the Web UI
  last_text: ""
  last_voice_mode: predefined
  last_predefined_voice: null
  last_reference_file: null
  last_seed: 42
  last_chunk_size: 120
  last_split_text_enabled: true
  hide_chunk_warning: false
  hide_generation_warning: false
```

### 11.3 Managing Configuration via UI

The Web UI provides sections to manage parts of the `config.yaml` file:

*   **Generation Parameters Section:** Sliders and the Seed input reflect values from `generation_defaults`. Clicking "Save Generation Parameters" updates this section in `config.yaml`.
*   **Server Configuration Section:** Allows viewing and editing `server`, `model`, and `paths` sections. Clicking "Save Server Configuration" updates `config.yaml`. Remember to restart the server for these changes to apply.
*   **UI State:** Text input, voice mode, file selections, chunking toggle/size are automatically saved to the `ui_state` section in `config.yaml` on change (debounced).

---

## 12. Running the Server

1.  **Activate Virtual Environment:** (`(venv)` prefix).
2.  **Navigate to Project Root:** (`dia-tts-server` directory).
3.  **(Optional First Run): Download Models:** Run `python download_model.py` to pre-download the configured Dia and Whisper models without starting the server. Monitor terminal output.
4.  **Run Server:**
    ```bash
    python server.py
    ```
5.  **Monitor Output:** Check logs for startup progress, `config.yaml` loading/creation, model loading (Dia, DAC, Whisper), device detection, and running status. If chunking is active, a `tqdm` progress bar appears during generation. Whisper transcription logs appear during cloning if needed.
6.  **Access:**
    *   **Web UI:** Should open automatically. If not, navigate to `http://localhost:PORT` (e.g., `http://localhost:8003`).
    *   **API Docs:** `http://localhost:PORT/docs`
7.  **Stop:** Press `CTRL+C` in the terminal.

---

## 13. Usage

### 13.1 Web User Interface (Web UI)

Access via the server's base URL (e.g., `http://localhost:8003`).

#### 13.1.1 Main Generation Form

*   **Text to speak:** Input text. Use `[S1]`/`[S2]` tags for dialogue and non-verbals `(like this)`. Content saved automatically to `config.yaml`.
*   **Generate Speech Button:** Initiates generation.

#### 13.1.2 Text Splitting / Chunking Controls

*   Controls automatic chunking (Section 3).
*   **Split text into chunks Checkbox:** Toggles chunking (default on). Saved automatically.
*   **Chunk Size Slider:** Sets target length for same-speaker aggregation (default 120). Saved automatically.
*   **Chunk Warning Dialog:** Warns if chunking is enabled with random seed (-1) in Random/Dialogue mode.

#### 13.1.3 Voice Mode Selection (Predefined, Clone, Random/Dialogue)

*   Select the desired voice generation method. Saved automatically.
*   **Predefined Voices:** Select from curated voices in the dropdown. Server uses corresponding audio/transcript from `./voices`. See Section 5.
*   **Voice Cloning:** Select reference file from dropdown (files in `./reference_audio`). Requires corresponding `.txt` transcript (recommended) or relies on Whisper fallback. Backend handles transcript prepending. See Section 4.
*   **Random Single / Dialogue:** Uses `[S1]`/`[S2]` tags if present, otherwise generates with a random S1 voice unless a fixed Seed is set.

#### 13.1.4 Presets

*   Load example text and parameters from `ui/presets.yaml`. Click buttons to apply. Customize by editing the YAML file.

#### 13.1.5 Generation Parameters (including Seeding)

*   Expandable section for Speed, CFG, Temp, Top P, Top K, Seed.
*   **Seed:** -1 for random, integer for specific results. Helps consistency with Predefined/Clone modes. Saved automatically.
*   "Save Generation Parameters" button updates `generation_defaults` in `config.yaml`.

#### 13.1.6 Server Configuration (UI)

*   Expandable section to view/edit `config.yaml` settings (Model Repo, Paths, Host, Port, Whisper Model). Requires server restart after saving.

#### 13.1.7 Generated Audio Player

*   Appears on success with waveform (WaveSurfer.js), play/pause, download button, and generation info.

#### 13.1.8 Theme Toggle

*   Switch between light/dark modes. Preference saved in browser local storage.

#### 13.1.9 Session Persistence

*   Text input, voice mode, file selections, generation parameters, chunking settings, and seed are saved in `config.yaml` (`ui_state` section) and restored on page load.

### 13.2 API Endpoints

Access interactive documentation (Swagger UI) via `/docs`.

#### 13.2.1 POST `/v1/audio/speech` (OpenAI Compatible)

*   **Purpose:** Provides an endpoint compatible with the basic OpenAI TTS API for easier integration with existing tools. Limited functionality compared to `/tts`.
*   **Request Body:** (`application/json`) - Uses the `OpenAITTSRequest` model.

    | Field             | Type                     | Required | Description                                                                                                                                                                                                                            | Default     |
    | :---------------- | :----------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- |
    | `model`           | string                   | No       | Ignored by this server (always uses Dia). Included for compatibility.                                                                                                                                                                  | `"dia-1.6b"`|
    | `input`           | string                   | Yes      | The text to synthesize. Use `[S1]`/`[S2]` tags for dialogue. **For cloning, you MUST manually prepend the reference transcript.**                                                                                                        |             |
    | `voice`           | string                   | No       | Maps to Dia modes. Use `"S1"`, `"S2"`, `"dialogue"`, the filename of a predefined voice (e.g., `"Michael_Emily.wav"`), or the filename of a reference audio (e.g., `"my_ref.wav"`) for cloning.                                        | `"S1"`      |
    | `response_format` | `"opus"` \| `"wav"`      | No       | Desired audio output format.                                                                                                                                                                                                           | `"opus"`    |
    | `speed`           | float                    | No       | Playback speed factor (0.5-2.0). Applied *after* generation.                                                                                                                                                                           | `1.0`       |
    | `seed`            | integer \| null          | No       | Generation seed. Use `-1` for random, or a specific integer for deterministic output.                                                                                                                                                  | `-1`        |

*   **Response:**
    *   **Success (200 OK):** `StreamingResponse` containing the binary audio data (`audio/opus` or `audio/wav`).
    *   **Error:** Standard FastAPI JSON error response (e.g., 400, 404, 500).
*   **Notes:** Implicitly uses chunking with default size. Does not support explicit transcript field or automatic transcript lookup/Whisper for cloning (manual prepending required).

#### 13.2.2 POST `/tts` (Custom Parameters)

*   **Purpose:** Offers full control over generation, including advanced cloning, chunking, and all parameters.
*   **Request Body:** (`application/json`) - Uses the `CustomTTSRequest` model.

    | Field                      | Type                                                 | Required | Description                                                                                                                                                                                                                                                        | Default     |
    | :------------------------- | :--------------------------------------------------- | :------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- |
    | `text`                     | string                                               | Yes      | The target text to synthesize. Do **not** manually prepend transcript for cloning here; use the `transcript` field or rely on automatic lookup.                                                                                                                   |             |
    | `voice_mode`               | `"dialogue"` \| `"single_s1"` \| `"single_s2"` \| `"clone"` \| `"predefined"` | No       | Specifies the generation mode.                                                                                                                                                                                                                     | `"single_s1"`|
    | `clone_reference_filename` | string \| null                                       | No       | Filename of the audio file in the corresponding path (`paths.reference_audio` for clone, `paths.voices` for predefined). **Required if `voice_mode` is `clone` or `predefined`**.                                                                                 | `null`      |
    | `transcript`               | string \| null                                       | No       | **(Clone Mode Only)** Explicit transcript of the reference audio. If provided, overrides local `.txt` file lookup and Whisper generation. Must include speaker tags (`[S1]`/`[S2]`).                                                                               | `null`      |
    | `output_format`            | `"opus"` \| `"wav"`                                  | No       | The desired audio output format.                                                                                                                                                                                                                                   | `"opus"`    |
    | `max_tokens`               | integer \| null                                      | No       | Maximum number of audio tokens to generate *per chunk*. `null` uses the model's internal default.                                                                                                                                                                  | `null`      |
    | `cfg_scale`                | float                                                | No       | Classifier-Free Guidance scale (1.0-5.0).                                                                                                                                                                                                                          | `3.0`       |
    | `temperature`              | float                                                | No       | Sampling temperature (0.1-1.5).                                                                                                                                                                                                                                  | `1.3`       |
    | `top_p`                    | float                                                | No       | Nucleus sampling probability (0.1-1.0).                                                                                                                                                                                                                          | `0.95`      |
    | `speed_factor`             | float                                                | No       | Adjusts the playback speed of the generated audio (0.5-2.0). Applied *after* generation.                                                                                                                                                                          | `1.0`       |
    | `cfg_filter_top_k`         | integer                                              | No       | Top k filter for CFG guidance (1-100).                                                                                                                                                                                                                             | `35`        |
    | `seed`                     | integer \| null                                      | No       | Generation seed. Use `-1` for random, or a specific integer for deterministic output.                                                                                                                                                                            | `42`        |
    | `split_text`               | boolean \| null                                      | No       | Enable/disable automatic text chunking.                                                                                                                                                                                                                            | `true`      |
    | `chunk_size`               | integer \| null                                      | No       | Approximate target character length for text chunks when splitting is enabled (100-1000 recommended).                                                                                                                                                            | `120`       |

*   **Response:**
    *   **Success (200 OK):** `StreamingResponse` containing the binary audio data (`audio/opus` or `audio/wav`).
    *   **Error:** Standard FastAPI JSON error response (e.g., 400, 404, 500).

#### 13.2.3 Configuration & Helper Endpoints

*   Primarily for UI use.
*   **POST `/save_settings`:** Saves partial updates to `config.yaml`.
*   **POST `/reset_settings`:** Resets `config.yaml` using defaults + `.env`.
*   **GET `/get_reference_files`:** Lists files in `paths.reference_audio`.
*   **GET `/get_predefined_voices`:** Lists formatted voices from `paths.voices`.
*   **POST `/upload_reference`:** Uploads reference audio files.
*   **GET `/health`:** Health check.

---

## 14. Troubleshooting

*   **CUDA Not Available / Slow:** Check drivers, PyTorch CUDA install (Section 10.4).
*   **VRAM Out of Memory (OOM):** Use BF16 model. Close other apps. Reduce `chunk_size` if using chunking. Whisper models also consume VRAM. Optimized code requires ~7GB.
*   **CUDA OOM During Startup:** Usually temporary overhead. Server loads weights to CPU first. If persists, check VRAM usage (`nvidia-smi`), ensure BF16 model, try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var.
*   **Import Errors (`dac`, `whisper`, `tqdm`, `yaml`, `parselmouth`):** Activate venv, run `pip install -r requirements.txt`.
*   **`libsndfile` / `ffmpeg` Error (Linux):** Install system libraries: `sudo apt update && sudo apt install libsndfile1 ffmpeg`.
*   **Model Download Fails (Dia or Whisper):** Check internet, `config.yaml` settings (`model.repo_id`, `model.weights_filename`, `model.whisper_model_name`), Hugging Face status, cache path permissions (`paths.model_cache`). Run `python download_model.py` manually for detailed logs.
*   **Voice Cloning Fails / Poor Quality:**
    *   **Transcript:** Ensure accurate `.txt` transcript exists in `./reference_audio` with correct `[S1]`/`[S2]` format. Whisper fallback is experimental.
    *   **Audio Quality:** Use clean reference audio (5-20s).
    *   **Logs:** Check server logs for errors from `_prepare_cloning_inputs`.
*   **Permission Errors (Saving Files/Config):** Check write permissions for `paths.output`, `paths.reference_audio`, `paths.voices`, `paths.model_cache` (for Whisper transcript saves), and `config.yaml`.
*   **UI Issues / Settings Not Saving:** Clear browser cache/local storage. Check developer console (F12). Ensure `config.yaml` is writable.
*   **Inconsistent Voice with Chunking:** Use "Predefined Voices" or "Voice Cloning" mode. If using "Random/Dialogue", use a fixed integer `seed` (not -1).
*   **Port Conflict (`Address already in use` / `Errno 98`):** Another process is using the port (default 8003). Stop the other process or change `server.port` in `config.yaml` (requires restart).
    *   **Explanation:** Usually caused by an unclean server shutdown or another app using the port.
    *   **Linux:** `sudo lsof -i:PORT | grep LISTEN | awk '{print $2}' | xargs kill -9` (Replace PORT).
    *   **Windows:** `for /f "tokens=5" %i in ('netstat -ano ^| findstr :PORT') do taskkill /F /PID %i` (Replace PORT). Use with caution.
*   **Generation Cancel Button:** This is a "UI Cancel" - it stops the *frontend* from waiting but doesn't instantly halt ongoing backend model inference.

---

## 15. Project Architecture

*   **`server.py`:** FastAPI app, API routes, UI routes (Jinja2), calls `engine.py`.
*   **`engine.py`:** Core logic. Holds Dia model instance. `generate_speech` orchestrates:
    *   Calls `utils._find_reference_file`.
    *   Calls `_prepare_cloning_inputs` for clone mode, handling audio processing and transcript acquisition (using `utils._generate_transcript_with_whisper` if needed).
    *   Calls `utils.chunk_text_by_sentences` for chunking.
    *   Applies seeding.
    *   Calls `dia_model.generate` (which handles the main TTS loop).
    *   Calls `utils` for audio post-processing.
*   **`config.py`:** Defines `YamlConfigManager`, default config structure, getter functions. Reads/writes `config.yaml`. Uses `.env` only for initial seeding/reset.
*   **`dia/` package:** Core Dia model implementation.
    *   `model.py`: `Dia` class wrapper, `generate` method implements inference loop.
    *   `layers.py`: PyTorch modules (Attention, MLP, etc.).
    *   `config.py`: Pydantic models for model architecture definition.
    *   `audio.py`: Low-level audio helpers (delay patterns).
    *   `state.py`: Classes for managing inference state (KV Cache).
*   **`ui/` directory:** Frontend files (HTML, JS, CSS via CDN, Presets YAML). JS handles UI state, API calls, player.
*   **`utils.py`:** Helper functions:
    *   Chunking logic (`chunk_text_by_sentences`, `split_into_sentences`, etc.).
    *   Whisper transcription (`_generate_transcript_with_whisper`).
    *   Audio utilities (`encode_audio`, `save_audio_to_file`, post-processing functions like `trim_lead_trail_silence`, `fix_internal_silence`, `remove_long_unvoiced_segments`).
    *   File handling (`sanitize_filename`, `get_valid_reference_files`, `get_predefined_voices`).
    *   Other (`PerformanceMonitor`).
*   **Dependencies:** FastAPI, Uvicorn, PyTorch, torchaudio, HF Hub, DAC, Whisper, soundfile, PyYAML, python-dotenv, pydantic, Jinja2, tqdm, requests, parselmouth.

---

## 16. License and Disclaimer

*   **License:** This project is licensed under the MIT License: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)
*   **Disclaimer:** This project offers a high-fidelity speech generation model intended primarily for research, educational, and creative use cases with synthetic or properly consented voices. The following uses are **strictly forbidden**:
    *   **Identity Misuse**: Do not produce audio resembling real individuals without explicit permission.
    *   **Deceptive Content**: Do not use this model to generate misleading content (e.g., fake news, impersonations).
    *   **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

    By using this model, you agree to uphold relevant legal standards and ethical responsibilities. The creators **are not responsible** for any misuse and firmly oppose any unethical usage of this technology. Prioritize using the included Predefined Voices or your own voice recordings for cloning to avoid ethical complications.
