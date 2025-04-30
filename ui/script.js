document.addEventListener('DOMContentLoaded', function () {
    // --- Global Flags ---
    let isGenerating = false;
    let isGenerationCancelled = false;
    let wavesurfer = null;
    let saveStateTimeout = null; // For debouncing state saves
    let hideChunkWarning = false; // Get initial warning flag state (will be updated in loadInitialState)    
    let hideGenerationWarning = false; // Add this line for the new warning

    // --- Element Selectors ---
    const ttsForm = document.getElementById('tts-form');
    const textArea = document.getElementById('text');
    const charCount = document.getElementById('char-count');
    const voiceModeRadios = document.querySelectorAll('input[name="voice_mode"]');
    const predefinedVoiceOptionsDiv = document.getElementById('predefined-voice-options');
    const predefinedVoiceSelect = document.getElementById('predefined_voice_select');
    const cloneOptionsDiv = document.getElementById('clone-options');
    const cloneReferenceSelect = document.getElementById('clone_reference_select');
    const cloneImportButton = document.getElementById('clone-import-button'); // Renamed
    const cloneRefreshButton = document.getElementById('clone-refresh-button'); // Added
    const cloneFileInput = document.getElementById('clone-file-input');
    const generateBtn = document.getElementById('generate-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const loadingStatus = document.getElementById('loading-status');
    const loadingCancelBtn = document.getElementById('loading-cancel-btn');
    const notificationArea = document.getElementById('notification-area');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const configSaveBtn = document.getElementById('save-config-btn');
    const configRestartBtn = document.getElementById('restart-server-btn');
    const configStatus = document.getElementById('config-status');
    const genDefaultsSaveBtn = document.getElementById('save-gen-defaults-btn');
    const genDefaultsStatus = document.getElementById('gen-defaults-status');
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const themeSwitchThumb = themeToggleButton ? themeToggleButton.querySelector('.theme-switch-thumb') : null;
    const presetsContainer = document.getElementById('presets-container');
    const splitTextToggle = document.getElementById('split-text-toggle');
    const chunkSizeControls = document.getElementById('chunk-size-controls');
    const chunkSizeSlider = document.getElementById('chunk-size-slider');
    const chunkSizeValue = document.getElementById('chunk-size-value');
    const chunkExplanation = document.getElementById('chunk-explanation');
    const seedInput = document.getElementById('seed');
    const resetSettingsBtn = document.getElementById('reset-settings-btn'); // Added
    const chunkWarningModal = document.getElementById('chunk-warning-modal');
    const chunkWarningOkBtn = document.getElementById('chunk-warning-ok');
    const chunkWarningCancelBtn = document.getElementById('chunk-warning-cancel');
    const hideChunkWarningCheckbox = document.getElementById('hide-chunk-warning-checkbox');
    const generationWarningModal = document.getElementById('generation-warning-modal');
    const generationWarningAcknowledgeBtn = document.getElementById('generation-warning-acknowledge');
    const hideGenerationWarningCheckbox = document.getElementById('hide-generation-warning-checkbox');

    // --- Configuration & State ---
    // Initial config loaded from server-rendered variable
    let currentConfig = window.initialConfig || {}; // Use loaded config or empty object
    let currentUiState = currentConfig.ui_state || {}; // Extract UI state
    // hideChunkWarning is initialized above, will be set in loadInitialState
    // hideGenerationWarning is initialized above, will be set in loadInitialState

    // --- Debounced State Saving ---
    const DEBOUNCE_DELAY = 750; // ms delay for saving state after changes

    async function saveCurrentUiState() {
        // Gather current UI state
        const stateToSave = {
            last_text: textArea ? textArea.value : '',
            last_voice_mode: document.querySelector('input[name="voice_mode"]:checked')?.value || 'predefined',
            last_predefined_voice: predefinedVoiceSelect ? predefinedVoiceSelect.value : null,
            last_reference_file: cloneReferenceSelect ? cloneReferenceSelect.value : null,
            last_seed: seedInput ? parseInt(seedInput.value) || 42 : 42,
            last_chunk_size: chunkSizeSlider ? parseInt(chunkSizeSlider.value) || 120 : 120,
            last_split_text_enabled: splitTextToggle ? splitTextToggle.checked : true,
            hide_chunk_warning: hideChunkWarning, // Use the JS variable state
            hide_generation_warning: hideGenerationWarning // Add the new flag state
        };

        console.log("Debounced save triggered. Saving UI state:", stateToSave);

        try {
            const response = await fetch('/save_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ui_state: stateToSave }) // Send only the ui_state part
            });
            if (!response.ok) {
                const errorResult = await response.json();
                throw new Error(errorResult.detail || `Failed to save UI state (status ${response.status})`);
            }
            console.log("UI state saved successfully.");
            // Optionally show a subtle saving indicator/confirmation
        } catch (error) {
            console.error("Error saving UI state:", error);
            // Optionally show a persistent error notification
            // showNotification(`Error saving settings: ${error.message}`, 'error', 0);
        }
    }

    function debouncedSaveState() {
        clearTimeout(saveStateTimeout);
        saveStateTimeout = setTimeout(saveCurrentUiState, DEBOUNCE_DELAY);
    }

    // --- Initial Setup ---

    // Character counter
    function updateCharCount() { if (textArea && charCount) charCount.textContent = textArea.value.length; }
    if (textArea) { textArea.addEventListener('input', updateCharCount); updateCharCount(); }

    // Toggle visibility of voice mode specific options
    function toggleVoiceOptions() {
        const selectedMode = document.querySelector('input[name="voice_mode"]:checked')?.value;
        const isPredefined = selectedMode === 'predefined';
        const isClone = selectedMode === 'clone';

        if (predefinedVoiceOptionsDiv && predefinedVoiceSelect) {
            predefinedVoiceOptionsDiv.classList.toggle('hidden', !isPredefined);
            predefinedVoiceSelect.required = isPredefined;
        }
        if (cloneOptionsDiv && cloneReferenceSelect && cloneImportButton && cloneRefreshButton) {
            cloneOptionsDiv.classList.toggle('hidden', !isClone);
            cloneReferenceSelect.required = isClone;
            // Buttons are always visible when clone mode is active
        }
        debouncedSaveState(); // Save state when mode changes
    }

    // Toggle visibility of Chunk Size Slider & Explanation
    function toggleChunkSlider() {
        const isChecked = splitTextToggle ? splitTextToggle.checked : false;
        if (chunkSizeControls) chunkSizeControls.classList.toggle('hidden', !isChecked);
        if (chunkExplanation) chunkExplanation.classList.toggle('hidden', !isChecked);
        debouncedSaveState(); // Save state when toggle changes
    }
    if (splitTextToggle) splitTextToggle.addEventListener('change', toggleChunkSlider);

    // Update Chunk Size Display
    function updateChunkSizeDisplay() {
        if (chunkSizeSlider && chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider.value;
        // No need to save state on every slider move, save on release (or use main form save)
    }
    if (chunkSizeSlider) {
        chunkSizeSlider.addEventListener('input', updateChunkSizeDisplay);
        chunkSizeSlider.addEventListener('change', debouncedSaveState); // Save on release
    }

    // Update slider value displays dynamically & save state on change
    const sliders = [
        { id: 'speed_factor', valueId: 'speed_factor_value' },
        { id: 'cfg_scale', valueId: 'cfg_scale_value' },
        { id: 'temperature', valueId: 'temperature_value' },
        { id: 'top_p', valueId: 'top_p_value' },
        { id: 'cfg_filter_top_k', valueId: 'cfg_filter_top_k_value' },
    ];
    sliders.forEach(sliderInfo => {
        const slider = document.getElementById(sliderInfo.id);
        const valueDisplay = document.getElementById(sliderInfo.valueId);
        if (slider && valueDisplay) {
            // Initial display is set by server template
            slider.addEventListener('input', () => valueDisplay.textContent = slider.value);
            slider.addEventListener('change', debouncedSaveState); // Save on release
        }
    });

    // Add listeners to save state for other inputs
    if (textArea) textArea.addEventListener('blur', debouncedSaveState); // Save text on blur
    if (seedInput) seedInput.addEventListener('change', debouncedSaveState);
    if (predefinedVoiceSelect) predefinedVoiceSelect.addEventListener('change', debouncedSaveState);
    if (cloneReferenceSelect) cloneReferenceSelect.addEventListener('change', debouncedSaveState);
    voiceModeRadios.forEach(radio => radio.addEventListener('change', toggleVoiceOptions)); // Already calls save

    // --- Load Initial State ---
    function loadInitialState() {
        console.log("Loading initial state from:", currentConfig);
        hideGenerationWarning = currentUiState.hide_generation_warning || false; // Load the new flag state
        hideChunkWarning = currentUiState.hide_chunk_warning || false; // Load the chunk warning flag state

        // Text Area
        if (textArea && currentUiState.last_text) {
            textArea.value = currentUiState.last_text;
            updateCharCount();
        }

        // Voice Mode
        const lastMode = currentUiState.last_voice_mode || 'predefined';
        const modeRadio = document.querySelector(`input[name="voice_mode"][value="${lastMode}"]`);
        if (modeRadio && !modeRadio.disabled) { // Ensure radio is not disabled (e.g., no predefined voices)
            modeRadio.checked = true;
        } else {
            // Fallback if saved mode is invalid or disabled
            const fallbackRadio = document.querySelector('input[name="voice_mode"]:not([disabled])');
            if (fallbackRadio) fallbackRadio.checked = true;
        }

        // Predefined Voice Selection
        if (predefinedVoiceSelect && currentUiState.last_predefined_voice) {
            // Check if the saved voice *filename* exists in the current options
            const voiceExists = Array.from(predefinedVoiceSelect.options).some(opt => opt.value === currentUiState.last_predefined_voice);
            if (voiceExists) {
                predefinedVoiceSelect.value = currentUiState.last_predefined_voice;
            } else {
                console.warn(`Saved predefined voice "${currentUiState.last_predefined_voice}" not found in current list. Resetting.`);
                predefinedVoiceSelect.value = 'none';
            }
        } else if (predefinedVoiceSelect) {
            predefinedVoiceSelect.value = 'none'; // Default if no value saved
        }

        // Reference File Selection
        if (cloneReferenceSelect && currentUiState.last_reference_file) {
            // Check if the saved reference file exists in the current options
            const refExists = Array.from(cloneReferenceSelect.options).some(opt => opt.value === currentUiState.last_reference_file);
            if (refExists) {
                cloneReferenceSelect.value = currentUiState.last_reference_file;
            } else {
                console.warn(`Saved reference file "${currentUiState.last_reference_file}" not found in current list. Resetting.`);
                cloneReferenceSelect.value = 'none';
            }
        } else if (cloneReferenceSelect) {
            cloneReferenceSelect.value = 'none'; // Default if no value saved
        }

        // Seed
        if (seedInput && currentUiState.last_seed !== undefined) {
            seedInput.value = currentUiState.last_seed;
        }

        // Split Text Toggle & Chunk Size
        if (splitTextToggle && currentUiState.last_split_text_enabled !== undefined) {
            splitTextToggle.checked = currentUiState.last_split_text_enabled;
        }
        if (chunkSizeSlider && currentUiState.last_chunk_size !== undefined) {
            chunkSizeSlider.value = currentUiState.last_chunk_size;
            updateChunkSizeDisplay();
        }

        // Apply initial visibility based on loaded state
        toggleVoiceOptions();
        toggleChunkSlider();

        // Handle default startup state if no text was loaded (implies first visit or reset)
        if (textArea && !textArea.value && typeof window.appPresets !== 'undefined' && Array.isArray(window.appPresets) && window.appPresets.length > 0) {
            console.log("Applying default startup preset (Short Dialogue) and voice.");
            const shortDialoguePreset = window.appPresets.find(p => p.name === "Short Dialogue");
            if (shortDialoguePreset) {
                applyPreset(shortDialoguePreset, false); // Apply preset without showing notification
            }

            // Override voice mode to predefined and select Michael-Emily or first available pair/single
            const predefinedRadio = document.querySelector('input[name="voice_mode"][value="predefined"]');
            if (predefinedRadio && !predefinedRadio.disabled) {
                predefinedRadio.checked = true;
                toggleVoiceOptions(); // Update UI visibility

                // Try to select predefined voice if available
                if (predefinedVoiceSelect && predefinedVoiceSelect.options.length > 1) {
                    // Try multiple filename variations due to possible underscore/hyphen inconsistency
                    const targetFiles = ["Michael_Emily.wav", "Michael-Emily.wav"];
                    let voiceSelected = false;

                    for (const filename of targetFiles) {
                        const targetOption = Array.from(predefinedVoiceSelect.options).find(opt => opt.value === filename);
                        if (targetOption) {
                            predefinedVoiceSelect.value = filename;
                            voiceSelected = true;
                            console.log(`Selected default voice: ${filename}`);
                            break;
                        }
                    }

                    // Fallback: Select first pair (contains '-') or first single
                    if (!voiceSelected) {
                        let fallbackOption = Array.from(predefinedVoiceSelect.options).find(opt => opt.textContent.includes('-')); // Find first pair
                        if (!fallbackOption) {
                            // If no pairs, use first option that's not the "-- Select Voice --" placeholder
                            const nonPlaceholderOptions = Array.from(predefinedVoiceSelect.options).filter(opt => opt.value !== 'none');
                            fallbackOption = nonPlaceholderOptions.length > 0 ? nonPlaceholderOptions[0] : null;
                        }

                        if (fallbackOption) {
                            predefinedVoiceSelect.value = fallbackOption.value;
                            console.log(`Selected fallback voice: ${fallbackOption.textContent}`);
                        }
                    }

                    debouncedSaveState(); // Save the default state
                }
            }
        } else if (textArea && !textArea.value) {
            // Log if default preset couldn't be applied because presets weren't loaded
            console.log("No initial text found, but presets are unavailable. Skipping default preset application.");
        }
    }

    // --- Notifications ---
    function showNotification(message, type = 'info', duration = 5000) {
        if (!notificationArea) return;
        const styles = {
            success: { base: 'notification-success', icon: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" /></svg>' },
            error: { base: 'notification-error', icon: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>' },
            warning: { base: 'notification-warning', icon: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" /></svg>' },
            info: { base: 'notification-info', icon: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd" /></svg>' }
        };
        const notificationDiv = document.createElement('div');
        const styleInfo = styles[type] || styles['info'];
        notificationDiv.className = styleInfo.base;
        notificationDiv.setAttribute('role', 'alert');
        notificationDiv.innerHTML = `${styleInfo.icon} <span class="block sm:inline">${message}</span>`;
        notificationArea.appendChild(notificationDiv);
        if (duration > 0) {
            setTimeout(() => {
                notificationDiv.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
                notificationDiv.style.opacity = '0';
                notificationDiv.style.transform = 'translateY(-10px)';
                setTimeout(() => notificationDiv.remove(), 500);
            }, duration);
        }
        return notificationDiv;
    }

    // --- Presets ---
    function applyPreset(presetData, showNotif = true) {
        console.log("Applying preset:", presetData);
        if (!presetData) {
            console.warn("Empty preset data provided to applyPreset");
            return;
        }

        // Apply text if available
        if (textArea && presetData.text !== undefined) {
            textArea.value = presetData.text;
            updateCharCount();
        } else {
            console.warn("Could not apply preset text: textArea not found or preset has no text");
        }

        // Apply params if available
        if (presetData.params) {
            console.log("Applying preset parameters:", presetData.params);
            for (const [key, value] of Object.entries(presetData.params)) {
                const slider = document.getElementById(key);
                const valueDisplay = document.getElementById(`${key}_value`);

                if (slider) {
                    console.log(`Setting ${key} to ${value}`);
                    slider.value = value;
                    if (valueDisplay) valueDisplay.textContent = value;
                } else if (key === 'seed' && seedInput) {
                    seedInput.value = value;
                } else {
                    console.warn(`Element not found for preset parameter: ${key}`);
                }
            }
        } else {
            console.warn("Preset has no params property");
        }

        if (showNotif) showNotification(`Preset "${presetData.name}" loaded.`, 'info', 3000);
        debouncedSaveState(); // Save state after applying preset
    }

    // Add event listeners to preset buttons if presets data is available
    if (typeof window.appPresets !== 'undefined' && Array.isArray(window.appPresets) && presetsContainer) {
        if (window.appPresets.length > 0) {
            window.appPresets.forEach((preset, index) => {
                const button = document.getElementById(`preset-btn-${index}`);
                if (button) {
                    button.addEventListener('click', () => applyPreset(preset));
                } else {
                    console.warn(`Preset button preset-btn-${index} not found.`);
                }
            });
            console.log(`Attached listeners to ${window.appPresets.length} preset buttons.`);
        } else {
            console.log("Presets data (window.appPresets) is an empty array. No preset listeners attached.");
            // Optionally display a message in the presetsContainer if it's empty
            // presetsContainer.innerHTML = '<p class="text-sm text-slate-500 dark:text-slate-400">No presets available.</p>';
        }
    } else if (presetsContainer) {
        // Log a warning if the variable is missing or not an array
        console.warn("Presets data (window.appPresets) is missing or invalid, preset buttons will not work.");
        // Optionally display a message
        // presetsContainer.innerHTML = '<p class="text-sm text-slate-500 dark:text-slate-400">Could not load presets.</p>';
    }

    // --- Audio Player ---
    function initializeWaveSurfer(audioUrl, resultDetails = {}) {
        if (wavesurfer) wavesurfer.destroy();
        const waveformDiv = document.getElementById('waveform');
        const playBtn = document.getElementById('play-btn');
        const durationSpan = document.getElementById('audio-duration');
        const playerContainer = document.getElementById('audio-player-container'); // Get container

        // Dynamically create player HTML if it doesn't exist (e.g., first generation)
        if (!waveformDiv && playerContainer) {
            playerContainer.innerHTML = `
                <div class="audio-player-card">
                    <div class="p-6 sm:p-8">
                        <h2 class="card-header">Generated Audio</h2>
                        <div class="mb-5"><div id="waveform" class="waveform-container"></div></div>
                        <div class="audio-player-controls">
                            <div class="audio-player-buttons">
                                <button id="play-btn" class="btn-primary" disabled>...</button>
                                <a id="download-link" href="#" download="" class="btn-secondary">...</a>
                            </div>
                            <div class="audio-player-info text-xs sm:text-sm">
                                Mode: <span id="player-voice-mode" class="font-medium text-indigo-600 dark:text-indigo-400">--</span>
                                <span id="player-voice-file-details"></span>
                                <span class="mx-1">•</span> Gen Time: <span id="player-gen-time" class="font-medium tabular-nums">--s</span>
                                <span class="mx-1">•</span> Duration: <span id="audio-duration" class="font-medium tabular-nums">--:--</span>
                            </div>
                        </div>
                    </div>
                </div>`;
            // Re-select elements after creation
            // waveformDiv = document.getElementById('waveform'); // Not needed again here
            // playBtn = document.getElementById('play-btn');
            // durationSpan = document.getElementById('audio-duration');
        } else if (!playerContainer) { // Only need to check the main container now
            console.error("Audio player container '#audio-player-container' not found.");
            // Optionally display an error message elsewhere if the container itself is missing
            return; // Cannot proceed without the container
        }
        // Re-select potentially dynamic elements inside the container *after* potential creation
        const currentWaveformDiv = document.getElementById('waveform');
        const currentPlayBtn = document.getElementById('play-btn');
        const currentDurationSpan = document.getElementById('audio-duration');

        // Check if essential inner elements exist after potential creation
        if (!currentWaveformDiv || !currentPlayBtn || !currentDurationSpan) {
            console.error("Essential audio player elements (waveform, play-btn, duration) not found within the container.");
            if (playerContainer) playerContainer.innerHTML = '<p class="text-red-500 dark:text-red-400 p-6">Error: Could not display audio player components.</p>';
            return;
        }


        // Update download link and info spans
        const downloadLink = document.getElementById('download-link');
        const playerModeSpan = document.getElementById('player-voice-mode');
        const playerFileSpan = document.getElementById('player-voice-file-details');
        const playerGenTimeSpan = document.getElementById('player-gen-time');
        const audioFilename = audioUrl.split('/').pop();

        if (downloadLink) {
            downloadLink.href = audioUrl;
            downloadLink.download = audioFilename || 'dia_tts_output.wav';
            downloadLink.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg> Download WAV`;
        }
        if (playerModeSpan) playerModeSpan.textContent = resultDetails.submittedVoiceMode || '--';
        if (playerFileSpan) {
            let fileDetail = '';
            if (resultDetails.submittedVoiceMode === 'clone' && resultDetails.submittedCloneFile) {
                fileDetail = `(<span class="font-medium text-slate-700 dark:text-slate-300">${resultDetails.submittedCloneFile}</span>)`;
            } else if (resultDetails.submittedVoiceMode === 'predefined' && resultDetails.submittedPredefinedVoice) {
                fileDetail = `(<span class="font-medium text-slate-700 dark:text-slate-300">${resultDetails.submittedPredefinedVoice}</span>)`;
            }
            playerFileSpan.innerHTML = fileDetail;
        }
        if (playerGenTimeSpan) playerGenTimeSpan.textContent = resultDetails.genTime ? `${resultDetails.genTime}s` : '--s';

        // Ensure buttons don't wrap excessively
        currentPlayBtn.classList.add('whitespace-nowrap', 'flex-shrink-0');
        if (downloadLink) downloadLink.classList.add('whitespace-nowrap', 'flex-shrink-0');

        const playIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg> Play`;
        const pauseIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm5-2.25A.75.75 0 0 1 7.75 7h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75v-4.5Z" clip-rule="evenodd" /></svg> Pause`;
        const isDarkMode = document.documentElement.classList.contains('dark');

        wavesurfer = WaveSurfer.create({
            container: currentWaveformDiv, // Use the potentially re-queried element
            waveColor: isDarkMode ? '#6366f1' : '#a5b4fc',
            progressColor: isDarkMode ? '#4f46e5' : '#6366f1',
            cursorColor: isDarkMode ? '#cbd5e1' : '#475569',
            barWidth: 3, barRadius: 3, cursorWidth: 1, height: 80, barGap: 2,
            responsive: true, url: audioUrl, mediaControls: false, normalize: true,
        });

        wavesurfer.on('ready', () => {
            const duration = wavesurfer.getDuration();
            const minutes = Math.floor(duration / 60);
            const seconds = Math.floor(duration % 60);
            currentDurationSpan.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
            currentPlayBtn.disabled = false;
            currentPlayBtn.innerHTML = playIconSVG;
        });
        wavesurfer.on('play', () => { currentPlayBtn.innerHTML = pauseIconSVG; });
        wavesurfer.on('pause', () => { currentPlayBtn.innerHTML = playIconSVG; });
        wavesurfer.on('finish', () => { currentPlayBtn.innerHTML = playIconSVG; wavesurfer.seekTo(0); });
        wavesurfer.on('error', (err) => {
            console.error("WaveSurfer error:", err);
            showNotification(`Error loading audio waveform: ${err}`, 'error');
            if (currentWaveformDiv) currentWaveformDiv.innerHTML = `<p class="p-4 text-sm text-red-600 dark:text-red-400">Could not load waveform.</p>`;
            currentPlayBtn.disabled = true;
        });

        currentPlayBtn.onclick = () => { wavesurfer.playPause(); };

        // --- *** ADDED SCROLLING LOGIC *** ---
        // Use setTimeout to ensure the element is rendered and layout is stable
        setTimeout(() => {
            // Re-select the container just in case, although it should exist
            const targetContainer = document.getElementById('audio-player-container');
            if (targetContainer) {
                console.log("Scrolling to audio player container...");
                targetContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                console.warn("Could not find #audio-player-container to scroll to.");
            }
        }, 150); // 150ms delay, adjust if needed
        // --- *** END OF ADDED SCROLLING LOGIC *** ---
    }

    // Initialize player if audio URL is present on initial page load
    if (window.initialGenResult && window.initialGenResult.outputUrl) {
        console.log("Initializing WaveSurfer for initially loaded audio.");
        initializeWaveSurfer(window.initialGenResult.outputUrl, window.initialGenResult);
    }

    // --- Form Submission & Loading State ---

    // --- Helper functions for submission flow (MOVED OUTSIDE LISTENER) ---
    function proceedWithSubmissionChecks() {
        // Check chunk warning conditions (reuse existing logic)
        const text = textArea.value.trim();
        const mode = document.querySelector('input[name="voice_mode"]:checked')?.value;
        const isSplitting = splitTextToggle && splitTextToggle.checked;
        const currentChunkSize = chunkSizeSlider ? parseInt(chunkSizeSlider.value) : 120;
        const needsChunkWarning = isSplitting &&
            text.length >= currentChunkSize * 2 &&
            mode === 'dialogue' && // Only show for random/dialogue mode
            !hideChunkWarning; // Check the chunk warning flag

        if (needsChunkWarning) {
            showChunkWarningModal();
            // Wait for chunk modal interaction (OK button calls showLoadingOverlayAndSubmit)
            return;
        }

        // If no chunk warning, proceed to final step
        showLoadingOverlayAndSubmit();
    }

    function showLoadingOverlayAndSubmit() {
        showLoadingOverlay(); // Show the spinner etc.
        ttsForm.submit(); // Explicitly submit the form now
    }
    // --- End Helper functions ---

    if (ttsForm) {
        ttsForm.addEventListener('submit', function (event) {
            event.preventDefault(); // Prevent default submission initially

            // --- Start of actual checks ---

            // Basic client-side validation (reuse existing logic)
            const text = textArea.value.trim();
            const mode = document.querySelector('input[name="voice_mode"]:checked')?.value;
            const predefinedFile = predefinedVoiceSelect?.value;
            const cloneFile = cloneReferenceSelect?.value;

            if (!text) {
                showNotification("Please enter some text to generate speech.", 'error');
                return; // Stop processing
            }
            if (mode === 'predefined' && (!predefinedFile || predefinedFile === 'none')) {
                showNotification("Please select a predefined voice.", 'error');
                return; // Stop processing
            }
            if (mode === 'clone' && (!cloneFile || cloneFile === 'none')) {
                showNotification("Please select a reference audio file for Voice Clone mode.", 'error');
                return; // Stop processing
            }

            // --- Check *General* Generation Warning ---
            if (!hideGenerationWarning) {
                showGenerationWarningModal();
                // Wait for generation warning modal interaction (Acknowledge button calls proceedWithSubmissionChecks)
                return;
            }

            // If general warning is already dismissed, proceed to next checks
            proceedWithSubmissionChecks();

        }); // --- End of ttsForm submit listener ---
    }


    function showLoadingOverlay() {
        if (isGenerating) {
            console.log("Generate clicked while previous generation in progress. Setting cancel flag.");
            showNotification("Cancelling previous request...", 'warning', 2000);
            isGenerationCancelled = true;
        }
        isGenerating = true;
        isGenerationCancelled = false;

        if (loadingOverlay && generateBtn && loadingCancelBtn) {
            const isSplitting = splitTextToggle && splitTextToggle.checked;
            const currentChunkSize = chunkSizeSlider ? parseInt(chunkSizeSlider.value) : 120;
            const mightSplit = isSplitting && textArea && textArea.value.length >= currentChunkSize * 2;
            loadingMessage.textContent = mightSplit ? 'Generating audio (processing chunks)...' : 'Generating audio...';
            loadingStatus.innerHTML = 'Please wait. This may take a moment. Check server terminal for detailed status.';

            loadingOverlay.classList.remove('hidden');
            requestAnimationFrame(() => {
                loadingOverlay.classList.remove('opacity-0');
                loadingOverlay.dataset.state = 'open';
            });
            generateBtn.disabled = true;
            loadingCancelBtn.disabled = false;
        }
    }

    function hideLoadingOverlay() {
        if (loadingOverlay && generateBtn) {
            loadingOverlay.classList.add('opacity-0');
            loadingOverlay.dataset.state = 'closed';
            setTimeout(() => loadingOverlay.classList.add('hidden'), 300);
            generateBtn.disabled = false;
        }
    }

    // Handle Cancel button click
    if (loadingCancelBtn) {
        loadingCancelBtn.addEventListener('click', () => {
            if (isGenerating) {
                console.log("Cancel button clicked.");
                isGenerationCancelled = true;
                isGenerating = false;
                hideLoadingOverlay();
                showNotification("Generation cancelled by user.", 'info');
            }
        });
    }

    // --- Chunk Warning Modal ---
    function showChunkWarningModal() {
        if (chunkWarningModal) {
            chunkWarningModal.classList.remove('hidden');
            requestAnimationFrame(() => chunkWarningModal.classList.remove('opacity-0'));
        }
    }
    function hideChunkWarningModal() {
        if (chunkWarningModal) {
            chunkWarningModal.classList.add('opacity-0');
            setTimeout(() => chunkWarningModal.classList.add('hidden'), 300); // Match transition duration
        }
    }

    if (chunkWarningOkBtn) {
        chunkWarningOkBtn.addEventListener('click', () => {
            if (hideChunkWarningCheckbox && hideChunkWarningCheckbox.checked) {
                hideChunkWarning = true; // Update JS flag
                // Save the updated flag to config.yaml via API
                fetch('/save_settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ui_state: { hide_chunk_warning: true } })
                }).catch(err => console.error("Failed to save hide chunk warning setting:", err));
            }
            hideChunkWarningModal();
            showLoadingOverlayAndSubmit(); // Call the helper to show overlay and submit
        });
    }
    if (chunkWarningCancelBtn) {
        chunkWarningCancelBtn.addEventListener('click', () => {
            hideChunkWarningModal();
            showNotification("Generation cancelled.", 'info');
        });
    }

    // --- General Generation Warning Modal ---
    function showGenerationWarningModal() {
        // const modal = document.getElementById('generation-warning-modal'); // Already selected globally
        if (generationWarningModal) {
            generationWarningModal.classList.remove('hidden');
            requestAnimationFrame(() => generationWarningModal.classList.remove('opacity-0'));
        }
    }
    function hideGenerationWarningModal() {
        // const modal = document.getElementById('generation-warning-modal'); // Already selected globally
        if (generationWarningModal) {
            generationWarningModal.classList.add('opacity-0');
            setTimeout(() => generationWarningModal.classList.add('hidden'), 300); // Match transition duration
        }
    }

    // Handle Generation Warning Acknowledge button
    // const generationWarningAcknowledgeBtn = document.getElementById('generation-warning-acknowledge'); // Already selected globally
    // const hideGenerationWarningCheckbox = document.getElementById('hide-generation-warning-checkbox'); // Already selected globally

    if (generationWarningAcknowledgeBtn && hideGenerationWarningCheckbox) {
        generationWarningAcknowledgeBtn.addEventListener('click', () => {
            if (hideGenerationWarningCheckbox.checked) {
                hideGenerationWarning = true; // Update JS flag
                // Save the updated flag state immediately
                // Note: We call the save function directly, not debounced, as it's a specific user action.
                saveCurrentUiState().catch(err => console.error("Failed immediate save of hide generation warning setting:", err));
            }
            hideGenerationWarningModal();

            // Manually trigger the next step in the submission flow
            proceedWithSubmissionChecks(); // Call the function directly to continue the flow

        });
    } else {
        console.warn("Could not find generation warning modal acknowledge button or checkbox.");
    }

    // --- Result Handling (on page load after form submission) ---
    if (window.initialGenResult && window.initialGenResult.outputUrl) {
        console.log("Page loaded with generation result URL:", window.initialGenResult.outputUrl);
        if (isGenerationCancelled) {
            console.log("Generation was cancelled before result loaded, ignoring.");
            showNotification("Previous generation was cancelled.", "warning");
            isGenerationCancelled = false;
        } else {
            console.log("Processing successful generation result.");
            initializeWaveSurfer(window.initialGenResult.outputUrl, window.initialGenResult);
        }
    } else {
        console.log("Page loaded without an initial audio result.");
    }
    isGenerating = false;
    if (generateBtn) generateBtn.disabled = false;
    if (loadingOverlay && loadingOverlay.dataset.state !== 'open') {
        loadingOverlay.classList.add('hidden', 'opacity-0');
        loadingOverlay.dataset.state = 'closed';
    }

    // --- Configuration Management ---
    async function updateConfigStatus(button, statusElement, message, type = 'info', duration = 5000) {
        const statusClasses = { success: 'text-green-600 dark:text-green-400', error: 'text-red-600 dark:text-red-400', warning: 'text-yellow-600 dark:text-yellow-400', info: 'text-indigo-600 dark:text-indigo-400', processing: 'text-yellow-600 dark:text-yellow-400 animate-pulse' };
        const messageType = (message.toLowerCase().includes('saving') || message.toLowerCase().includes('restarting') || message.toLowerCase().includes('resetting')) ? 'processing' : type;
        statusElement.textContent = message;
        statusElement.className = `text-xs ml-2 ${statusClasses[messageType] || statusClasses['info']}`;
        statusElement.classList.remove('hidden');
        if (button && messageType !== 'info') button.disabled = true;
        if (duration > 0) {
            setTimeout(() => {
                statusElement.classList.add('hidden');
                if (button) button.disabled = false;
            }, duration);
        } else if (duration === 0 && type === 'error') {
            if (button) button.disabled = false;
        }
    }

    // Save Server Configuration
    if (configSaveBtn) {
        configSaveBtn.addEventListener('click', async () => {
            const configData = {};
            const serverConfigInputs = document.querySelectorAll('#server-config-form input[name]');
            serverConfigInputs.forEach(input => {
                const keys = input.name.split('.'); // e.g., "server.port" -> ["server", "port"]
                let currentLevel = configData;
                keys.forEach((key, index) => {
                    if (index === keys.length - 1) {
                        // Try converting known numeric types
                        let value = input.value;
                        if (input.type === 'number') {
                            value = input.name.includes('port') ? parseInt(value) : parseFloat(value);
                        }
                        currentLevel[key] = value;
                    } else {
                        currentLevel[key] = currentLevel[key] || {};
                        currentLevel = currentLevel[key];
                    }
                });
            });

            updateConfigStatus(configSaveBtn, configStatus, 'Saving configuration...', 'info', 0);
            try {
                const response = await fetch('/save_settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(configData) // Send nested structure
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save configuration');
                updateConfigStatus(configSaveBtn, configStatus, result.message, 'success', 5000);
                if (result.restart_needed && configRestartBtn) configRestartBtn.classList.remove('hidden');
                configSaveBtn.disabled = false;
            } catch (error) {
                console.error('Error saving server config:', error);
                updateConfigStatus(configSaveBtn, configStatus, `Error: ${error.message}`, 'error', 0);
            }
        });
    }

    // Restart Server
    if (configRestartBtn) {
        configRestartBtn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to restart the server? Ensure your process manager handles restarts.")) return;
            configRestartBtn.disabled = true;
            if (configSaveBtn) configSaveBtn.disabled = true;
            configRestartBtn.innerHTML = `<svg class="animate-spin h-5 w-5 mr-1.5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Restarting...`;
            updateConfigStatus(null, configStatus, 'Attempting server restart...', 'info', 0);
            try {
                const response = await fetch('/restart_server', { method: 'POST' }); // Assuming /restart_server still exists for this action
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Server responded with error on restart command');
                updateConfigStatus(null, configStatus, result.message + " Waiting for server...", 'info', 0);
                if (loadingOverlay) { /* Show main loading overlay */
                    loadingMessage.textContent = 'Server restarting...';
                    loadingStatus.textContent = 'Waiting for server to respond...';
                    loadingCancelBtn.disabled = true;
                    loadingOverlay.classList.remove('hidden');
                    requestAnimationFrame(() => { loadingOverlay.classList.remove('opacity-0'); loadingOverlay.dataset.state = 'open'; });
                }
                // --- Polling Logic ---
                let attempts = 0; const maxAttempts = 60; const pollInterval = 1000;
                function checkServerReady() {
                    attempts++; console.log(`Checking server readiness (Attempt ${attempts}/${maxAttempts})...`);
                    if (loadingStatus) loadingStatus.textContent = `Waiting for server... (${attempts}/${maxAttempts})`;
                    fetch('/health?cache=' + Date.now(), { cache: 'no-store', headers: { 'pragma': 'no-cache', 'cache-control': 'no-cache' } })
                        .then(res => res.ok ? res.json() : Promise.reject(res.status))
                        .then(healthData => {
                            if (healthData.model_loaded) {
                                console.log("Server ready. Reloading page.");
                                if (loadingStatus) loadingStatus.textContent = "Server ready! Reloading page...";
                                setTimeout(() => window.location.reload(true), 1500);
                            } else if (attempts < maxAttempts) {
                                console.log("Server responding but model not loaded yet, retrying...");
                                if (loadingStatus) loadingStatus.textContent = `Server up, loading model... (${attempts}/${maxAttempts})`;
                                setTimeout(checkServerReady, pollInterval);
                            } else { throw new Error('Server restarted but model failed to load within timeout.'); }
                        }).catch((err) => {
                            console.warn("Health check failed:", err);
                            if (attempts < maxAttempts) { setTimeout(checkServerReady, pollInterval); }
                            else { throw new Error('Server did not respond after restart.'); }
                        });
                }
                setTimeout(checkServerReady, 3000);
            } catch (error) {
                console.error('Error restarting server:', error);
                updateConfigStatus(null, configStatus, `Restart Error: ${error.message}`, 'error', 0);
                configRestartBtn.disabled = false;
                if (configSaveBtn) configSaveBtn.disabled = false;
                configRestartBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-1.5 inline-block"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" /></svg> Restart Server`;
                if (loadingOverlay) { /* Hide loading overlay on error */
                    loadingOverlay.classList.add('opacity-0'); loadingOverlay.dataset.state = 'closed';
                    setTimeout(() => loadingOverlay.classList.add('hidden'), 300);
                }
            }
        });
    }

    // Save Generation Defaults
    if (genDefaultsSaveBtn) {
        genDefaultsSaveBtn.addEventListener('click', async () => {
            const genParams = {};
            sliders.forEach(s => { const slider = document.getElementById(s.id); if (slider) genParams[s.id] = slider.type === 'range' ? parseFloat(slider.value) : slider.value; });
            if (seedInput) genParams['seed'] = parseInt(seedInput.value) || 42;
            if (splitTextToggle) genParams['split_text'] = splitTextToggle.checked;
            if (chunkSizeSlider) genParams['chunk_size'] = parseInt(chunkSizeSlider.value) || 120;

            updateConfigStatus(genDefaultsSaveBtn, genDefaultsStatus, 'Saving defaults...', 'info', 0);
            try {
                const response = await fetch('/save_settings', { // Use the unified save endpoint
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ generation_defaults: genParams }) // Send nested structure
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save generation defaults');
                updateConfigStatus(genDefaultsSaveBtn, genDefaultsStatus, result.message, 'success', 5000);
                genDefaultsSaveBtn.disabled = false;
            } catch (error) {
                console.error('Error saving generation defaults:', error);
                updateConfigStatus(genDefaultsSaveBtn, genDefaultsStatus, `Error: ${error.message}`, 'error', 0);
            }
        });
    }

    // Reset All Settings
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to reset ALL settings (including UI state, generation defaults, and server config) back to their default values (potentially overridden by .env)? This cannot be undone.")) {
                return;
            }
            updateConfigStatus(resetSettingsBtn, configStatus, 'Resetting settings...', 'info', 0); // Use config status area
            try {
                const response = await fetch('/reset_settings', { method: 'POST' });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to reset settings');
                updateConfigStatus(resetSettingsBtn, configStatus, 'Settings reset. Reloading page...', 'success', 0);
                setTimeout(() => window.location.reload(true), 1500); // Reload after showing message
            } catch (error) {
                console.error('Error resetting settings:', error);
                updateConfigStatus(resetSettingsBtn, configStatus, `Reset Error: ${error.message}`, 'error', 0);
            }
        });
    }


    // --- Reference Audio Upload & Refresh ---
    if (cloneImportButton && cloneFileInput && cloneReferenceSelect) {
        cloneImportButton.addEventListener('click', () => cloneFileInput.click());
        cloneFileInput.addEventListener('change', async (event) => {
            const files = event.target.files; if (!files || files.length === 0) return;
            cloneImportButton.disabled = true;
            cloneImportButton.innerHTML = `<svg class="animate-spin h-5 w-5 mr-1.5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Uploading...`;
            const uploadNotification = showNotification(`Uploading ${files.length} file(s)...`, 'info', 0);
            const formData = new FormData();
            for (const file of files) formData.append('files', file);
            try {
                const response = await fetch('/upload_reference', { method: 'POST', body: formData });
                const result = await response.json();
                if (uploadNotification) uploadNotification.remove();
                if (!response.ok) throw new Error(result.message || result.detail || `Upload failed`);
                if (result.errors && result.errors.length > 0) result.errors.forEach(err => showNotification(`Upload Warning: ${err}`, 'warning'));
                if (result.uploaded_files && result.uploaded_files.length > 0) showNotification(`Successfully uploaded: ${result.uploaded_files.join(', ')}`, 'success');
                else if (!result.errors || result.errors.length === 0) showNotification("Files processed. No new valid files added.", 'info');
                // Update dropdown
                const currentSelection = cloneReferenceSelect.value;
                cloneReferenceSelect.innerHTML = '<option value="none">-- Select Reference File --</option>';
                result.all_reference_files.forEach(filename => {
                    const option = document.createElement('option');
                    option.value = filename; option.textContent = filename;
                    cloneReferenceSelect.appendChild(option);
                });
                const firstUploaded = result.uploaded_files ? result.uploaded_files[0] : null;
                if (firstUploaded && result.all_reference_files.includes(firstUploaded)) cloneReferenceSelect.value = firstUploaded;
                else if (result.all_reference_files.includes(currentSelection)) cloneReferenceSelect.value = currentSelection;
                else cloneReferenceSelect.value = 'none';
                debouncedSaveState(); // Save the potentially updated selection
            } catch (error) {
                console.error('Error uploading reference files:', error);
                if (uploadNotification) uploadNotification.remove();
                showNotification(`Upload Error: ${error.message}`, 'error');
            } finally {
                cloneImportButton.disabled = false;
                cloneImportButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path d="M9.25 13.25a.75.75 0 0 0 1.5 0V4.636l2.955 3.129a.75.75 0 0 0 1.09-1.03l-4.25-4.5a.75.75 0 0 0-1.09 0l-4.25 4.5a.75.75 0 1 0 1.09 1.03L9.25 4.636v8.614Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg> Import`;
                cloneFileInput.value = '';
            }
        });
    }

    if (cloneRefreshButton && cloneReferenceSelect) {
        cloneRefreshButton.addEventListener('click', async () => {
            cloneRefreshButton.disabled = true;
            const originalIcon = cloneRefreshButton.innerHTML;
            cloneRefreshButton.innerHTML = `<svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
            try {
                const response = await fetch('/get_reference_files');
                if (!response.ok) throw new Error('Failed to fetch reference files');
                const files = await response.json();
                const currentSelection = cloneReferenceSelect.value;
                cloneReferenceSelect.innerHTML = '<option value="none">-- Select Reference File --</option>';
                files.forEach(filename => {
                    const option = document.createElement('option');
                    option.value = filename; option.textContent = filename;
                    cloneReferenceSelect.appendChild(option);
                });
                if (files.includes(currentSelection)) cloneReferenceSelect.value = currentSelection;
                else cloneReferenceSelect.value = 'none';
                showNotification("Reference file list refreshed.", 'info', 2000);
                debouncedSaveState(); // Save potentially changed selection
            } catch (error) {
                console.error("Error refreshing reference files:", error);
                showNotification(`Error refreshing list: ${error.message}`, 'error');
            } finally {
                cloneRefreshButton.disabled = false;
                cloneRefreshButton.innerHTML = originalIcon;
            }
        });
    }


    // --- Theme Toggle ---
    function applyTheme(theme) {
        const isDark = theme === 'dark';
        document.documentElement.classList.toggle('dark', isDark);
        if (themeSwitchThumb) {
            themeSwitchThumb.classList.toggle('translate-x-6', isDark); // Adjust based on your button/thumb size
            themeSwitchThumb.classList.toggle('bg-indigo-500', isDark);
            themeSwitchThumb.classList.toggle('bg-white', !isDark);
        }
        // Update wavesurfer colors
        if (wavesurfer && wavesurfer.isReady) {
            wavesurfer.setOptions({
                waveColor: isDark ? '#6366f1' : '#a5b4fc',
                progressColor: isDark ? '#4f46e5' : '#6366f1',
                cursorColor: isDark ? '#cbd5e1' : '#475569',
            });
        }
    }
    if (themeToggleButton) {
        // Use theme from config if available, else default to dark
        const initialTheme = currentConfig?.ui_state?.theme || 'dark'; // Assuming theme might be saved in future
        applyTheme(initialTheme);
        themeToggleButton.addEventListener('click', () => {
            const isCurrentlyDark = document.documentElement.classList.contains('dark');
            const newTheme = isCurrentlyDark ? 'light' : 'dark';
            applyTheme(newTheme);
            // Save theme preference (optional, could add to ui_state)
            // fetch('/save_settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ui_state: { theme: newTheme } }) });
            console.log("Theme changed to:", newTheme);
        });
    }

    // --- Initial Load ---
    loadInitialState(); // Load state from config after setting up listeners

}); // End DOMContentLoaded
