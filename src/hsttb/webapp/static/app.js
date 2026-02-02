// HSTTB Web Application JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // ========================================================================
    // DOM Elements
    // ========================================================================

    // Text inputs
    const groundTruthInput = document.getElementById('ground-truth');
    const predictedInput = document.getElementById('predicted');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsSection = document.getElementById('results-section');
    const examplesContainer = document.getElementById('examples-container');

    // Metric checkboxes
    const computeWer = document.getElementById('compute-wer');
    const computeCer = document.getElementById('compute-cer');
    const computeTer = document.getElementById('compute-ter');
    const computeNer = document.getElementById('compute-ner');
    const computeCrs = document.getElementById('compute-crs');
    const computeQuality = document.getElementById('compute-quality');
    const multiBackendToggle = document.getElementById('multi-backend');
    const multiNlpToggle = document.getElementById('multi-nlp');
    const nlpModelSection = document.getElementById('nlp-model-section');
    const multiBackendDescription = document.getElementById('multi-backend-description');

    // Audio elements
    const audioDropzone = document.getElementById('audio-dropzone');
    const audioFileInput = document.getElementById('audio-file-input');
    const audioPreview = document.getElementById('audio-preview');
    const audioPlayer = document.getElementById('audio-player');
    const audioFilename = document.getElementById('audio-filename');
    const audioDuration = document.getElementById('audio-duration');
    const removeAudioBtn = document.getElementById('remove-audio');
    const transcribeBtn = document.getElementById('transcribe-btn');
    const adapterSection = document.getElementById('adapter-section');

    // Recording elements
    const recordBtn = document.getElementById('record-btn');
    const recordingTimer = document.getElementById('recording-timer');
    const recordingStatus = document.getElementById('recording-status');

    // TTS elements
    const ttsText = document.getElementById('tts-text');
    const ttsVoice = document.getElementById('tts-voice');
    const ttsLabel = document.getElementById('tts-label');
    const ttsGenerateBtn = document.getElementById('tts-generate-btn');
    const ttsResult = document.getElementById('tts-result');
    const ttsAudioPlayer = document.getElementById('tts-audio-player');
    const ttsUseAudioBtn = document.getElementById('tts-use-audio');
    const ttsError = document.getElementById('tts-error');
    const ttsErrorMessage = document.getElementById('tts-error-message');

    // TTS History elements
    const ttsHistoryList = document.getElementById('tts-history-list');
    const ttsHistoryEmpty = document.getElementById('tts-history-empty');
    const historyCount = document.getElementById('history-count');
    const historySize = document.getElementById('history-size');
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // Tab elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Adapter elements
    const adapterCards = document.querySelectorAll('.adapter-card');
    const whisperModel = document.getElementById('whisper-model');

    // State
    let currentAudioFileId = null;
    let currentAudioFile = null;
    let lastResults = null;
    let radarChart = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordingStartTime = null;
    let recordingInterval = null;
    let ttsGeneratedFileId = null;
    let ttsHistoryEntries = [];

    // Additional elements
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const headerStatus = document.getElementById('header-status');
    const scoreDescription = document.getElementById('score-description');

    // ========================================================================
    // Initialization
    // ========================================================================

    loadExamples();
    loadBackends();
    loadTTSHistory();
    loadNLPModels();
    setupEventListeners();
    setupCheckboxLabels();
    setupInfoButtons();

    // ========================================================================
    // Event Listeners
    // ========================================================================

    function setupEventListeners() {
        // Evaluation
        evaluateBtn.addEventListener('click', runEvaluation);
        clearBtn.addEventListener('click', clearForm);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                runEvaluation();
            }
        });

        // Multi-Backend toggle
        multiBackendToggle.addEventListener('change', () => {
            if (multiBackendDescription) {
                multiBackendDescription.classList.toggle('hidden', !multiBackendToggle.checked);
            }
        });

        // Multi-NLP toggle
        multiNlpToggle.addEventListener('change', () => {
            nlpModelSection.classList.toggle('hidden', !multiNlpToggle.checked);
        });

        // Tabs
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => switchTab(btn.dataset.tab));
        });

        // Audio dropzone
        setupDropzone();

        // Adapter selection
        adapterCards.forEach(card => {
            card.addEventListener('click', () => selectAdapter(card));
        });

        // Transcribe button
        transcribeBtn.addEventListener('click', transcribeAudio);

        // Remove audio
        removeAudioBtn.addEventListener('click', removeAudio);

        // Recording
        recordBtn.addEventListener('click', toggleRecording);

        // TTS
        if (ttsGenerateBtn) {
            ttsGenerateBtn.addEventListener('click', generateTTS);
        }
        if (ttsUseAudioBtn) {
            ttsUseAudioBtn.addEventListener('click', useTTSAudio);
        }

        // TTS History
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener('click', loadTTSHistory);
        }
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', clearTTSHistory);
        }

        // Export buttons
        document.getElementById('export-json')?.addEventListener('click', exportJSON);
        document.getElementById('export-csv')?.addEventListener('click', exportCSV);
    }

    function setupCheckboxLabels() {
        // Add checked class to checkbox labels based on their state
        const checkboxLabels = document.querySelectorAll('.checkbox-label input[type="checkbox"]');
        checkboxLabels.forEach(checkbox => {
            const label = checkbox.closest('.checkbox-label');
            // Set initial state
            if (checkbox.checked) {
                label.classList.add('checked');
            }
            // Listen for changes
            checkbox.addEventListener('change', () => {
                label.classList.toggle('checked', checkbox.checked);
            });
        });
    }

    // Metric information for info buttons
    const metricInfo = {
        wer: {
            title: 'Word Error Rate (WER)',
            content: `
                <h4>What it measures</h4>
                <p>WER measures the edit distance between the reference (ground truth) and hypothesis (predicted) transcriptions at the <strong>word level</strong>.</p>

                <h4>Formula</h4>
                <div class="formula">WER = (S + D + I) / N</div>
                <ul>
                    <li><strong>S</strong> = Number of word substitutions</li>
                    <li><strong>D</strong> = Number of word deletions</li>
                    <li><strong>I</strong> = Number of word insertions</li>
                    <li><strong>N</strong> = Total words in reference</li>
                </ul>

                <h4>Interpretation</h4>
                <p>Lower WER is better. A WER of 0% means perfect transcription. WER can exceed 100% if there are many insertions.</p>
                <ul>
                    <li><strong>0-5%</strong>: Excellent</li>
                    <li><strong>5-10%</strong>: Good</li>
                    <li><strong>10-20%</strong>: Acceptable</li>
                    <li><strong>&gt;20%</strong>: Poor</li>
                </ul>
            `
        },
        cer: {
            title: 'Character Error Rate (CER)',
            content: `
                <h4>What it measures</h4>
                <p>CER measures the edit distance at the <strong>character level</strong>. More granular than WER, it's useful for detecting spelling errors and typos.</p>

                <h4>Formula</h4>
                <div class="formula">CER = (S + D + I) / N</div>
                <ul>
                    <li><strong>S</strong> = Number of character substitutions</li>
                    <li><strong>D</strong> = Number of character deletions</li>
                    <li><strong>I</strong> = Number of character insertions</li>
                    <li><strong>N</strong> = Total characters in reference</li>
                </ul>

                <h4>When to use</h4>
                <p>CER is particularly useful for:</p>
                <ul>
                    <li>Detecting minor spelling errors (e.g., "metformin" vs "metforman")</li>
                    <li>Languages without clear word boundaries</li>
                    <li>Evaluating character-level accuracy of drug names and dosages</li>
                </ul>
            `
        },
        ter: {
            title: 'Term Error Rate (TER)',
            content: `
                <h4>What it measures</h4>
                <p>TER is a <strong>healthcare-specific metric</strong> that measures the accuracy of medical term transcription. Unlike WER, it focuses only on clinically important terms.</p>

                <h4>How it works</h4>
                <ul>
                    <li>Extracts medical terms (drugs, conditions, procedures, dosages) from both texts</li>
                    <li>Compares extracted terms using fuzzy matching</li>
                    <li>Calculates error rate based on term-level substitutions, deletions, and insertions</li>
                </ul>

                <h4>Formula</h4>
                <div class="formula">TER = (S + D + I) / Total Medical Terms</div>

                <h4>Why it matters</h4>
                <p>In healthcare, some errors are more critical than others:</p>
                <ul>
                    <li><strong>Drug name errors</strong>: "metformin" → "methotrexate" (critical!)</li>
                    <li><strong>Dosage errors</strong>: "500mg" → "50mg" (dangerous)</li>
                    <li><strong>Common word errors</strong>: "the" → "a" (less important)</li>
                </ul>
                <p>TER weights medical terms appropriately for clinical safety.</p>
            `
        },
        ner: {
            title: 'Named Entity Recognition (NER)',
            content: `
                <h4>What it measures</h4>
                <p>NER evaluates how well medical entities (drugs, conditions, procedures) are preserved in the transcription.</p>

                <h4>Metrics</h4>
                <ul>
                    <li><strong>Precision</strong>: Of entities in prediction, how many are correct?</li>
                    <li><strong>Recall</strong>: Of entities in ground truth, how many were found?</li>
                    <li><strong>F1 Score</strong>: Harmonic mean of precision and recall</li>
                </ul>

                <h4>Formulas</h4>
                <div class="formula">Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (P × R) / (P + R)</div>

                <h4>Additional metrics</h4>
                <ul>
                    <li><strong>Entity Distortion Rate</strong>: Entities that were incorrectly transcribed</li>
                    <li><strong>Entity Omission Rate</strong>: Entities that were completely missed</li>
                </ul>
            `
        },
        crs: {
            title: 'Context Retention Score (CRS)',
            content: `
                <h4>What it measures</h4>
                <p>CRS evaluates whether the <strong>overall meaning and context</strong> is preserved, even if individual words differ.</p>

                <h4>Components</h4>
                <ul>
                    <li><strong>Semantic Similarity</strong>: How similar is the meaning? (using sentence embeddings)</li>
                    <li><strong>Entity Continuity</strong>: Are medical entities consistently mentioned?</li>
                    <li><strong>Negation Consistency</strong>: Are negations preserved? ("no pain" vs "pain")</li>
                    <li><strong>Context Drift</strong>: Does meaning shift across segments?</li>
                </ul>

                <h4>Why it matters</h4>
                <p>Some transcription errors change meaning completely:</p>
                <ul>
                    <li>"Patient has <strong>no</strong> chest pain" → "Patient has chest pain" (negation flip!)</li>
                    <li>"Discontinue medication" → "Continue medication" (critical error)</li>
                </ul>
                <p>CRS catches these context-changing errors that WER might miss.</p>
            `
        },
        quality: {
            title: 'Quality Score (Reference-Free)',
            content: `
                <h4>What it measures</h4>
                <p>Quality Score evaluates transcription quality <strong>without needing ground truth</strong>. Useful when you only have the transcription.</p>

                <h4>Core Components</h4>
                <ul>
                    <li><strong>Perplexity</strong>: Language model fluency (GPT-2). Lower = more natural text.</li>
                    <li><strong>Grammar</strong>: Grammatical correctness (rule-based checker)</li>
                    <li><strong>Entity Validity</strong>: Are medical terms spelled correctly and valid?</li>
                    <li><strong>Coherence</strong>: Do drug-condition pairs make clinical sense?</li>
                </ul>

                <h4>Advanced Components</h4>
                <ul>
                    <li><strong>Contradiction Detection</strong>: Finds internal contradictions (e.g., "denies diabetes" then "diabetes medication")</li>
                    <li><strong>Semantic Stability</strong>: Measures meaning consistency across transcript segments using embeddings</li>
                    <li><strong>Confidence Variance</strong>: Analyzes next-word probability stability to detect garbled text</li>
                </ul>

                <h4>Recommendation</h4>
                <ul>
                    <li><strong>ACCEPT</strong> (≥75%): Good quality, likely accurate</li>
                    <li><strong>REVIEW</strong> (50-75%): Manual review recommended</li>
                    <li><strong>REJECT</strong> (&lt;50%): Poor quality, likely contains errors</li>
                </ul>

                <h4>Limitations</h4>
                <p>Quality score can't detect factual errors - a fluent, grammatical sentence with wrong drug names will still score high. Always use with reference-based metrics when ground truth is available.</p>
            `
        },
        'multi-backend': {
            title: 'Multi-Backend Comparison',
            content: `
                <h4>What it does</h4>
                <p>Compares different text matching algorithms for calculating Term Error Rate (TER).</p>

                <h4>Available backends</h4>
                <ul>
                    <li><strong>RapidFuzz</strong>: Fast fuzzy string matching, good for handling typos</li>
                    <li><strong>Difflib</strong>: Python's built-in sequence matcher</li>
                    <li><strong>Basic</strong>: Simple word-by-word exact comparison</li>
                </ul>

                <h4>Why compare?</h4>
                <p>Different algorithms may yield slightly different TER scores:</p>
                <ul>
                    <li>Fuzzy matching may be more forgiving of minor spelling differences</li>
                    <li>Exact matching is stricter but may overcount errors</li>
                </ul>
                <p>Comparing backends helps understand the robustness of your TER measurement.</p>
            `
        }
    };

    function setupInfoButtons() {
        const infoModal = document.getElementById('info-modal');
        const infoModalTitle = document.getElementById('info-modal-title');
        const infoModalBody = document.getElementById('info-modal-body');
        const infoModalClose = document.getElementById('info-modal-close');

        if (!infoModal) return;

        // Handle info button clicks
        document.querySelectorAll('.info-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();

                const metric = btn.dataset.metric;
                const info = metricInfo[metric];

                if (info) {
                    infoModalTitle.textContent = info.title;
                    infoModalBody.innerHTML = info.content;
                    infoModal.classList.add('active');
                }
            });
        });

        // Close modal
        infoModalClose.addEventListener('click', () => {
            infoModal.classList.remove('active');
        });

        // Close on overlay click
        infoModal.addEventListener('click', (e) => {
            if (e.target === infoModal) {
                infoModal.classList.remove('active');
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && infoModal.classList.contains('active')) {
                infoModal.classList.remove('active');
            }
        });
    }

    function updateHeaderStatus(status, text) {
        if (!headerStatus) return;
        headerStatus.className = 'header-status ' + status;
        const statusText = headerStatus.querySelector('.status-text');
        if (statusText) {
            statusText.textContent = text;
        }
    }

    // ========================================================================
    // Tab Management
    // ========================================================================

    function switchTab(tabId) {
        tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tabId}`);
        });

        // Show/hide adapter section based on tab
        if (tabId === 'text' || tabId === 'tts' || tabId === 'history') {
            adapterSection.classList.add('hidden');
        } else {
            adapterSection.classList.remove('hidden');
        }

        // Refresh history when switching to history tab
        if (tabId === 'history') {
            loadTTSHistory();
        }
    }

    // ========================================================================
    // Audio Upload
    // ========================================================================

    function setupDropzone() {
        audioDropzone.addEventListener('click', () => audioFileInput.click());

        audioDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            audioDropzone.classList.add('dragover');
        });

        audioDropzone.addEventListener('dragleave', () => {
            audioDropzone.classList.remove('dragover');
        });

        audioDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            audioDropzone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) handleAudioFile(file);
        });

        audioFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleAudioFile(file);
        });
    }

    async function handleAudioFile(file) {
        // Validate file type
        const validTypes = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!validTypes.includes(ext)) {
            alert('Invalid file type. Supported: ' + validTypes.join(', '));
            return;
        }

        currentAudioFile = file;

        // Show preview
        audioFilename.textContent = file.name;
        audioPlayer.src = URL.createObjectURL(file);
        audioDropzone.classList.add('hidden');
        audioPreview.classList.remove('hidden');
        transcribeBtn.classList.remove('hidden');

        // Get duration when loaded
        audioPlayer.addEventListener('loadedmetadata', () => {
            const duration = formatDuration(audioPlayer.duration);
            audioDuration.textContent = duration;
        });

        // Upload file
        await uploadAudioFile(file);
    }

    async function uploadAudioFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/audio/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (data.status === 'success') {
                currentAudioFileId = data.file_id;
                console.log('Audio uploaded:', currentAudioFileId);
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload audio: ' + error.message);
        }
    }

    function removeAudio() {
        currentAudioFile = null;
        currentAudioFileId = null;
        audioPlayer.src = '';
        audioPreview.classList.add('hidden');
        audioDropzone.classList.remove('hidden');
        transcribeBtn.classList.add('hidden');
        audioFileInput.value = '';
    }

    // ========================================================================
    // Recording
    // ========================================================================

    async function toggleRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            recordedChunks = [];

            mediaRecorder.addEventListener('dataavailable', (e) => {
                if (e.data.size > 0) {
                    recordedChunks.push(e.data);
                }
            });

            mediaRecorder.addEventListener('stop', async () => {
                const blob = new Blob(recordedChunks, { type: 'audio/webm' });
                const file = new File([blob], 'recording.webm', { type: 'audio/webm' });

                // Switch to upload tab and handle the file
                switchTab('upload');
                await handleAudioFile(file);

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            });

            mediaRecorder.start();
            recordingStartTime = Date.now();

            // Update UI
            recordBtn.classList.add('recording');
            recordBtn.querySelector('.record-text').textContent = 'Stop Recording';
            recordingTimer.classList.remove('hidden');
            recordingStatus.textContent = 'Recording...';

            // Start timer
            recordingInterval = setInterval(updateRecordingTimer, 100);

        } catch (error) {
            console.error('Recording error:', error);
            recordingStatus.textContent = 'Microphone access denied';
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }

        // Update UI
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Start Recording';
        recordingStatus.textContent = '';

        // Stop timer
        clearInterval(recordingInterval);
        recordingTimer.classList.add('hidden');
    }

    function updateRecordingTimer() {
        const elapsed = (Date.now() - recordingStartTime) / 1000;
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = Math.floor(elapsed % 60).toString().padStart(2, '0');
        recordingTimer.textContent = `${mins}:${secs}`;
    }

    // ========================================================================
    // TTS Generation
    // ========================================================================

    async function generateTTS() {
        const text = ttsText.value.trim();

        if (!text) {
            showTTSError('Please enter text to convert to speech.');
            return;
        }

        const voice = ttsVoice.value;
        const label = ttsLabel ? ttsLabel.value.trim() : '';

        // Show loading state
        ttsGenerateBtn.disabled = true;
        ttsGenerateBtn.classList.add('loading');
        ttsResult.classList.add('hidden');
        ttsError.classList.add('hidden');

        try {
            const response = await fetch('/api/tts/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    voice: voice,
                    label: label,
                    save_to_history: true,
                }),
            });

            const data = await response.json();

            if (data.status === 'success') {
                ttsGeneratedFileId = data.file_id;

                // Set audio player source
                ttsAudioPlayer.src = `/api/audio/file/${data.file_id}`;

                // Also set the ground truth text
                groundTruthInput.value = text;

                // Show success result
                ttsResult.classList.remove('hidden');
                ttsError.classList.add('hidden');

                // Clear label input after successful generation
                if (ttsLabel) {
                    ttsLabel.value = '';
                }
            } else {
                throw new Error(data.error || 'TTS generation failed');
            }
        } catch (error) {
            console.error('TTS error:', error);
            showTTSError(error.message);
        } finally {
            ttsGenerateBtn.disabled = false;
            ttsGenerateBtn.classList.remove('loading');
        }
    }

    function showTTSError(message) {
        ttsErrorMessage.textContent = message;
        ttsError.classList.remove('hidden');
        ttsResult.classList.add('hidden');
    }

    async function useTTSAudio() {
        if (!ttsGeneratedFileId) {
            alert('No TTS audio generated yet.');
            return;
        }

        // Set as current audio file
        currentAudioFileId = ttsGeneratedFileId;

        // Copy text to ground truth
        if (ttsText.value.trim()) {
            groundTruthInput.value = ttsText.value.trim();
        }

        // Switch to upload tab to show audio controls
        switchTab('upload');

        // Show transcribe button
        transcribeBtn.classList.remove('hidden');
        adapterSection.classList.remove('hidden');

        // Update audio preview
        audioFilename.textContent = 'TTS Generated Audio';
        audioPlayer.src = ttsAudioPlayer.src;
        audioDropzone.classList.add('hidden');
        audioPreview.classList.remove('hidden');

        // Scroll to adapter section
        adapterSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // ========================================================================
    // TTS History Management
    // ========================================================================

    async function loadTTSHistory() {
        try {
            const response = await fetch('/api/tts/history');
            const data = await response.json();

            if (data.status === 'success') {
                ttsHistoryEntries = data.entries;
                renderTTSHistory();
                updateHistoryStats(data.stats);
            }
        } catch (error) {
            console.error('Failed to load TTS history:', error);
        }
    }

    function updateHistoryStats(stats) {
        if (historyCount) {
            historyCount.textContent = stats.count;
        }
        if (historySize) {
            historySize.textContent = stats.total_size_mb.toFixed(1);
        }
    }

    function renderTTSHistory() {
        if (!ttsHistoryList) return;

        // Clear existing entries (except the empty state)
        const existingEntries = ttsHistoryList.querySelectorAll('.tts-history-entry');
        existingEntries.forEach(entry => entry.remove());

        // Show/hide empty state
        if (ttsHistoryEntries.length === 0) {
            if (ttsHistoryEmpty) ttsHistoryEmpty.classList.remove('hidden');
            return;
        }

        if (ttsHistoryEmpty) ttsHistoryEmpty.classList.add('hidden');

        // Render entries
        ttsHistoryEntries.forEach(entry => {
            const entryEl = createHistoryEntryElement(entry);
            ttsHistoryList.appendChild(entryEl);
        });
    }

    function createHistoryEntryElement(entry) {
        const div = document.createElement('div');
        div.className = 'tts-history-entry';
        div.dataset.fileId = entry.file_id;

        const date = new Date(entry.created_at);
        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const textPreview = entry.text.length > 60 ? entry.text.substring(0, 57) + '...' : entry.text;

        div.innerHTML = `
            <div class="history-entry-header">
                <span class="history-entry-label">${entry.label || 'Untitled'}</span>
                <span class="history-entry-voice">${entry.voice}</span>
            </div>
            <div class="history-entry-text">${escapeHtml(textPreview)}</div>
            <div class="history-entry-footer">
                <span class="history-entry-date">${formattedDate}</span>
                <span class="history-entry-size">${(entry.file_size / 1024).toFixed(0)} KB</span>
            </div>
            <div class="history-entry-actions">
                <button class="btn-history-play" title="Play audio">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                </button>
                <button class="btn-history-use" title="Use for evaluation">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Use
                </button>
                <button class="btn-history-delete" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            </div>
            <audio class="history-entry-audio" src="/api/tts/history/${entry.file_id}/audio"></audio>
        `;

        // Event listeners
        const playBtn = div.querySelector('.btn-history-play');
        const useBtn = div.querySelector('.btn-history-use');
        const deleteBtn = div.querySelector('.btn-history-delete');
        const audio = div.querySelector('.history-entry-audio');

        playBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            // Stop other audio
            document.querySelectorAll('.history-entry-audio').forEach(a => {
                if (a !== audio) {
                    a.pause();
                    a.currentTime = 0;
                }
            });
            if (audio.paused) {
                audio.play();
                playBtn.classList.add('playing');
            } else {
                audio.pause();
                playBtn.classList.remove('playing');
            }
        });

        audio.addEventListener('ended', () => {
            playBtn.classList.remove('playing');
        });

        useBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            useHistoryEntry(entry);
        });

        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteHistoryEntry(entry.file_id);
        });

        return div;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async function useHistoryEntry(entry) {
        // Set as current audio file
        currentAudioFileId = entry.file_id;
        ttsGeneratedFileId = entry.file_id;

        // Set ground truth from the stored text
        groundTruthInput.value = entry.text;

        // Switch to upload tab to show audio controls
        switchTab('upload');

        // Show transcribe button
        transcribeBtn.classList.remove('hidden');
        adapterSection.classList.remove('hidden');

        // Update audio preview
        audioFilename.textContent = entry.label || 'TTS Audio';
        audioPlayer.src = `/api/tts/history/${entry.file_id}/audio`;
        audioDropzone.classList.add('hidden');
        audioPreview.classList.remove('hidden');

        // Scroll to adapter section
        adapterSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    async function deleteHistoryEntry(fileId) {
        if (!confirm('Delete this TTS audio entry?')) {
            return;
        }

        try {
            const response = await fetch(`/api/tts/history/${fileId}`, {
                method: 'DELETE',
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Remove from local array and re-render
                ttsHistoryEntries = ttsHistoryEntries.filter(e => e.file_id !== fileId);
                renderTTSHistory();

                // Reload stats
                loadTTSHistory();
            } else {
                alert('Failed to delete entry: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Failed to delete history entry:', error);
            alert('Failed to delete entry');
        }
    }

    async function clearTTSHistory() {
        if (!confirm('Delete ALL TTS history entries? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch('/api/tts/history', {
                method: 'DELETE',
            });

            const data = await response.json();

            if (data.status === 'success') {
                ttsHistoryEntries = [];
                renderTTSHistory();
                updateHistoryStats({ count: 0, total_size_mb: 0 });
            } else {
                alert('Failed to clear history: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Failed to clear history:', error);
            alert('Failed to clear history');
        }
    }

    // ========================================================================
    // Adapter Selection
    // ========================================================================

    function selectAdapter(card) {
        adapterCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
    }

    function getSelectedAdapter() {
        const selected = document.querySelector('.adapter-card.selected');
        return selected ? selected.dataset.adapter : 'whisper';
    }

    // ========================================================================
    // Transcription
    // ========================================================================

    async function transcribeAudio() {
        if (!currentAudioFileId) {
            alert('Please upload an audio file first.');
            return;
        }

        const adapter = getSelectedAdapter();
        const model = adapter === 'whisper' ? whisperModel.value : null;

        transcribeBtn.disabled = true;
        transcribeBtn.classList.add('loading');

        try {
            const response = await fetch('/api/audio/transcribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audio_file_id: currentAudioFileId,
                    adapter: adapter,
                    model: model,
                }),
            });

            const data = await response.json();

            if (data.status === 'success') {
                predictedInput.value = data.transcript;
                predictedInput.scrollIntoView({ behavior: 'smooth', block: 'center' });

                // Display speech rate analysis if available
                if (data.speech_rate) {
                    displaySpeechRate(data.speech_rate);
                }
            } else {
                throw new Error(data.error || 'Transcription failed');
            }
        } catch (error) {
            console.error('Transcription error:', error);
            alert('Transcription failed: ' + error.message);
        } finally {
            transcribeBtn.disabled = false;
            transcribeBtn.classList.remove('loading');
        }
    }

    function displaySpeechRate(speechRate) {
        const section = document.getElementById('speech-rate-section');
        if (!section) return;

        section.classList.remove('hidden');

        // Update WPM
        const wpmEl = document.getElementById('speech-rate-wpm');
        if (wpmEl) wpmEl.textContent = Math.round(speechRate.words_per_minute);

        // Update category
        const categoryEl = document.getElementById('speech-rate-category');
        if (categoryEl) {
            const categoryLabels = {
                'implausibly_low': 'Missing Content',
                'slow': 'Slow',
                'normal': 'Normal',
                'fast': 'Fast',
                'implausibly_high': 'Possible Hallucination'
            };
            categoryEl.textContent = categoryLabels[speechRate.category] || speechRate.category;
            categoryEl.className = 'speech-rate-category ' + speechRate.category;
        }

        // Update bar
        const barEl = document.getElementById('speech-rate-bar');
        if (barEl) barEl.style.width = (speechRate.plausibility_score * 100) + '%';

        // Update score
        const scoreEl = document.getElementById('speech-rate-score');
        if (scoreEl) scoreEl.textContent = Math.round(speechRate.plausibility_score * 100) + '%';

        // Update stats
        const wordsEl = document.getElementById('speech-rate-words');
        if (wordsEl) wordsEl.textContent = speechRate.word_count;

        const durationEl = document.getElementById('speech-rate-duration');
        if (durationEl) durationEl.textContent = Math.round(speechRate.audio_duration_seconds);

        // Show warning if any
        const warningEl = document.getElementById('speech-rate-warning');
        const warningTextEl = document.getElementById('speech-rate-warning-text');
        if (speechRate.warning) {
            warningEl.classList.remove('hidden');
            warningTextEl.textContent = speechRate.warning;
        } else {
            warningEl.classList.add('hidden');
        }

        // Show results section
        document.getElementById('results-placeholder').classList.add('hidden');
        document.getElementById('results-section').classList.remove('hidden');
    }

    // ========================================================================
    // Evaluation
    // ========================================================================

    async function runEvaluation() {
        const groundTruth = groundTruthInput.value.trim();
        const predicted = predictedInput.value.trim();
        const qualityEnabled = computeQuality && computeQuality.checked;

        // Predicted text is always required
        if (!predicted) {
            alert('Please enter the transcription text to evaluate.');
            return;
        }

        // If no ground truth and quality is not enabled, prompt user
        if (!groundTruth && !qualityEnabled) {
            alert('Please enter ground truth text, or enable "Quality Score" for reference-free evaluation.');
            return;
        }

        evaluateBtn.disabled = true;
        evaluateBtn.textContent = 'Evaluating...';
        evaluateBtn.classList.add('loading');
        updateHeaderStatus('processing', 'Evaluating...');

        try {
            // Run the base evaluation
            const baseResults = await runSingleEvaluation(groundTruth, predicted, qualityEnabled);

            // Additionally run multi-model comparisons if enabled (requires ground truth)
            if (groundTruth && multiNlpToggle.checked) {
                await runMultiNLPEvaluation(groundTruth, predicted, baseResults);
            }
            if (groundTruth && multiBackendToggle.checked) {
                await runMultiBackendEvaluation(groundTruth, predicted, baseResults);
            }
        } catch (error) {
            console.error('Evaluation error:', error);
            alert('Failed to run evaluation. Please try again.');
            updateHeaderStatus('error', 'Error');
        } finally {
            evaluateBtn.disabled = false;
            evaluateBtn.textContent = 'Evaluate Transcription';
            evaluateBtn.classList.remove('loading');
        }
    }

    async function runSingleEvaluation(groundTruth, predicted, qualityEnabled) {
        const requestBody = {
            predicted: predicted,
            compute_wer: groundTruth ? computeWer.checked : false,
            compute_cer: groundTruth ? computeCer.checked : false,
            compute_ter: groundTruth ? computeTer.checked : false,
            compute_ner: groundTruth ? computeNer.checked : false,
            compute_crs: groundTruth ? computeCrs.checked : false,
            compute_quality: qualityEnabled || false,
        };

        console.log('Evaluation request:', requestBody);

        // Only include ground_truth if provided
        if (groundTruth) {
            requestBody.ground_truth = groundTruth;
        }

        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        const results = await response.json();

        if (results.status === 'success') {
            lastResults = results;
            displayResults(results);
            if (groundTruth) {
                displayDiff(groundTruth, predicted);
            } else {
                // Hide diff section when no ground truth
                const diffSection = document.getElementById('diff-section');
                if (diffSection) diffSection.classList.add('hidden');
            }
            return results;
        } else {
            alert('Evaluation failed: ' + (results.error || 'Unknown error'));
            throw new Error(results.error || 'Evaluation failed');
        }
    }

    async function runMultiBackendEvaluation(groundTruth, predicted, baseResults) {
        const response = await fetch('/api/evaluate/multi-backend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ground_truth: groundTruth,
                predicted: predicted,
            }),
        });

        const results = await response.json();

        if (results.status === 'success') {
            // Merge with base results
            lastResults = { ...baseResults, multiBackend: results };
            displayMultiBackendResults(results);
            // Don't call displayDiff again - already shown by base evaluation
        } else {
            console.error('Multi-backend evaluation failed:', results.error);
        }
    }

    async function runMultiNLPEvaluation(groundTruth, predicted, baseResults) {
        // Get selected models
        const modelCheckboxes = document.querySelectorAll('#nlp-model-grid input:checked');
        const models = Array.from(modelCheckboxes).map(cb => cb.value);

        const response = await fetch('/api/evaluate/multi-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ground_truth: groundTruth,
                predicted: predicted,
                models: models.length > 0 ? models : null,
            }),
        });

        const results = await response.json();

        if (results.status === 'success') {
            // Merge with base results
            lastResults = { ...baseResults, multiNLP: results };
            displayMultiNLPResults(results);
            // Don't call displayDiff again - already shown by base evaluation
        } else {
            console.error('Multi-NLP evaluation failed:', results.error);
        }
    }

    // ========================================================================
    // Results Display
    // ========================================================================

    function displayResults(results) {
        // Hide placeholder, show results
        if (resultsPlaceholder) {
            resultsPlaceholder.classList.add('hidden');
        }
        resultsSection.classList.remove('hidden');

        // Determine evaluation mode
        const hasGroundTruth = results.ter || results.ner || results.crs;
        const hasQuality = results.quality;
        const isQualityOnly = results.mode === 'quality_only';

        // Hide multi-sections initially (they'll be shown if multi-evaluation runs)
        if (!multiNlpToggle.checked) {
            document.getElementById('multi-nlp-section')?.classList.add('hidden');
        }
        if (!multiBackendToggle.checked) {
            document.getElementById('multi-backend-section')?.classList.add('hidden');
        }
        if (!multiNlpToggle.checked && !multiBackendToggle.checked) {
            document.getElementById('chart-container')?.classList.add('hidden');
        }

        // Show/hide diff section based on whether we have ground truth
        const diffSection = document.getElementById('diff-section');
        if (diffSection) {
            diffSection.classList.toggle('hidden', isQualityOnly);
        }

        // Overall score
        const overallScoreValue = document.getElementById('overall-score-value');
        const overallScoreCircle = document.getElementById('overall-score-circle');

        if (results.overall_score !== undefined) {
            const score = results.overall_score;
            overallScoreValue.textContent = formatPercent(score);
            overallScoreCircle.className = 'score-circle ' + getScoreClass(score);

            // Update score description for non-technical users
            if (scoreDescription) {
                if (isQualityOnly) {
                    // Quality-only mode descriptions
                    if (score >= 0.85) {
                        scoreDescription.textContent = 'Good transcription quality. Text appears fluent and medical terms are valid.';
                    } else if (score >= 0.6) {
                        scoreDescription.textContent = 'Moderate quality. Some issues detected. Review recommended.';
                    } else {
                        scoreDescription.textContent = 'Poor quality. Significant issues with fluency or medical terms.';
                    }
                } else {
                    // Reference-based mode descriptions
                    if (score >= 0.95) {
                        scoreDescription.textContent = 'Excellent! The transcription is highly accurate with minimal errors. Safe for clinical use.';
                    } else if (score >= 0.85) {
                        scoreDescription.textContent = 'Good accuracy. Minor errors detected. Review recommended before clinical use.';
                    } else if (score >= 0.7) {
                        scoreDescription.textContent = 'Moderate accuracy. Some medical terms may be incorrect. Manual review required.';
                    } else {
                        scoreDescription.textContent = 'Poor accuracy. Significant errors detected. Not suitable for clinical use without correction.';
                    }
                }
            }
        } else {
            overallScoreValue.textContent = '--';
            overallScoreCircle.className = 'score-circle';
            if (scoreDescription) {
                scoreDescription.textContent = 'This score represents how accurately the speech-to-text system transcribed medical content.';
            }
        }

        // Update header status
        updateHeaderStatus('', 'Ready');

        // WER Results
        displayWERResults(results);

        // CER Results
        displayCERResults(results);

        // TER Results (hide if quality-only mode)
        displayTERResults(results);

        // NER Results (hide if quality-only mode)
        displayNERResults(results);

        // CRS Results (hide if quality-only mode)
        displayCRSResults(results);

        // Quality Results
        displayQualityResults(results);

        // Errors
        displayErrors(results);

        // Scroll to results - on desktop, results panel is already visible
        // On mobile, scroll to the results section
        if (window.innerWidth <= 1200) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function displayWERResults(results) {
        const werCard = document.getElementById('wer-card');
        const werValue = document.getElementById('wer-value');
        const werDetails = document.getElementById('wer-details');

        if (results.wer) {
            werCard.classList.remove('hidden');
            const werAccuracy = results.wer.word_accuracy;
            werValue.textContent = formatPercent(results.wer.wer);
            werValue.className = 'metric-value ' + getScoreClass(werAccuracy);
            werDetails.innerHTML = `
                <p><strong>Word Accuracy:</strong> ${formatPercent(werAccuracy)}</p>
                <p><strong>Reference Words:</strong> ${results.wer.reference_words}</p>
                <p><strong>Substitutions:</strong> ${results.wer.substitutions}</p>
                <p><strong>Deletions:</strong> ${results.wer.deletions}</p>
                <p><strong>Insertions:</strong> ${results.wer.insertions}</p>
            `;
        } else {
            werCard.classList.add('hidden');
        }
    }

    function displayCERResults(results) {
        const cerCard = document.getElementById('cer-card');
        const cerValue = document.getElementById('cer-value');
        const cerDetails = document.getElementById('cer-details');

        if (results.cer) {
            cerCard.classList.remove('hidden');
            const cerAccuracy = results.cer.char_accuracy;
            cerValue.textContent = formatPercent(results.cer.cer);
            cerValue.className = 'metric-value ' + getScoreClass(cerAccuracy);
            cerDetails.innerHTML = `
                <p><strong>Char Accuracy:</strong> ${formatPercent(cerAccuracy)}</p>
                <p><strong>Reference Chars:</strong> ${results.cer.reference_chars}</p>
                <p><strong>Substitutions:</strong> ${results.cer.substitutions}</p>
                <p><strong>Deletions:</strong> ${results.cer.deletions}</p>
                <p><strong>Insertions:</strong> ${results.cer.insertions}</p>
            `;
        } else {
            cerCard.classList.add('hidden');
        }
    }

    function displayTERResults(results) {
        const terCard = document.getElementById('ter-card');
        const terValue = document.getElementById('ter-value');
        const terDetails = document.getElementById('ter-details');

        if (results.ter) {
            terCard.classList.remove('hidden');
            const terScore = 1 - results.ter.overall_ter;
            terValue.textContent = formatPercent(results.ter.overall_ter);
            terValue.className = 'metric-value ' + getScoreClass(terScore);
            terDetails.innerHTML = `
                <p><strong>Total Terms:</strong> ${results.ter.total_terms}</p>
                <p><strong>Substitutions:</strong> ${results.ter.substitutions}</p>
                <p><strong>Deletions:</strong> ${results.ter.deletions}</p>
                <p><strong>Insertions:</strong> ${results.ter.insertions}</p>
            `;
        } else {
            terCard.classList.add('hidden');
        }
    }

    function displayNERResults(results) {
        const nerCard = document.getElementById('ner-card');
        const nerValue = document.getElementById('ner-value');
        const nerDetails = document.getElementById('ner-details');

        if (results.ner) {
            nerCard.classList.remove('hidden');
            nerValue.textContent = formatPercent(results.ner.f1_score);
            nerValue.className = 'metric-value ' + getScoreClass(results.ner.f1_score);
            nerDetails.innerHTML = `
                <p><strong>Precision:</strong> ${formatPercent(results.ner.precision)}</p>
                <p><strong>Recall:</strong> ${formatPercent(results.ner.recall)}</p>
                <p><strong>F1 Score:</strong> ${formatPercent(results.ner.f1_score)}</p>
                <p><strong>Distortion Rate:</strong> ${formatPercent(results.ner.entity_distortion_rate)}</p>
            `;
        } else {
            nerCard.classList.add('hidden');
        }
    }

    function displayCRSResults(results) {
        const crsCard = document.getElementById('crs-card');
        const crsValue = document.getElementById('crs-value');
        const crsDetails = document.getElementById('crs-details');

        if (results.crs) {
            crsCard.classList.remove('hidden');
            crsValue.textContent = formatPercent(results.crs.composite_score);
            crsValue.className = 'metric-value ' + getScoreClass(results.crs.composite_score);
            crsDetails.innerHTML = `
                <p><strong>Semantic Similarity:</strong> ${formatPercent(results.crs.semantic_similarity)}</p>
                <p><strong>Entity Continuity:</strong> ${formatPercent(results.crs.entity_continuity)}</p>
                <p><strong>Negation Consistency:</strong> ${formatPercent(results.crs.negation_consistency)}</p>
                <p><strong>Context Drift:</strong> ${formatPercent(results.crs.context_drift_rate)}</p>
            `;
        } else {
            crsCard.classList.add('hidden');
        }
    }

    function displayQualityResults(results) {
        const qualitySection = document.getElementById('quality-section');

        if (!qualitySection) return;

        if (!results.quality) {
            qualitySection.classList.add('hidden');
            return;
        }

        qualitySection.classList.remove('hidden');
        const quality = results.quality;

        // Update recommendation badge
        const recommendationEl = document.getElementById('quality-recommendation');
        if (recommendationEl) {
            recommendationEl.textContent = quality.recommendation;
            recommendationEl.className = 'quality-recommendation ' + quality.recommendation.toLowerCase();
        }

        // Update composite score
        const compositeValue = document.getElementById('quality-composite-value');
        if (compositeValue) {
            compositeValue.textContent = formatPercent(quality.composite_score);
            compositeValue.className = 'quality-score-value ' + getScoreClass(quality.composite_score);
        }

        // Update component scores
        updateQualityComponent('perplexity', quality.perplexity_score, quality.perplexity_available !== false);
        updateQualityComponent('grammar', quality.grammar_score, quality.grammar_available !== false);
        updateQualityComponent('entity', quality.entity_validity_score, true);
        updateQualityComponent('coherence', quality.coherence_score, true);

        // Update new metric scores
        updateQualityComponent('contradiction', quality.contradiction_score, true);
        updateQualityComponent('embedding-drift', quality.embedding_drift_score, quality.embedding_drift_available !== false);
        updateQualityComponent('confidence', quality.confidence_variance_score, quality.confidence_variance_available !== false);

        // Display entities found
        displayQualityEntities(quality.entities_found || []);

        // Display grammar issues
        displayGrammarIssues(quality.grammar_errors || []);

        // Display invalid entities
        displayInvalidEntities(quality.invalid_entities || []);

        // Display contradictions
        displayContradictions(quality.contradictions || []);

        // Display embedding drift points
        displayDriftPoints(quality.drift_points || []);

        // Display confidence drop points
        displayConfidenceDrops(quality.confidence_drop_points || []);
    }

    function updateQualityComponent(name, score, available) {
        const valueEl = document.getElementById(`quality-${name}-value`);
        const barEl = document.getElementById(`quality-${name}-bar`);

        if (valueEl) {
            if (available && score !== undefined) {
                valueEl.textContent = formatPercent(score);
                valueEl.className = 'component-value ' + getScoreClass(score);
            } else {
                valueEl.textContent = 'N/A';
                valueEl.className = 'component-value';
            }
        }

        if (barEl) {
            if (available && score !== undefined) {
                barEl.style.width = `${score * 100}%`;
                barEl.className = 'component-bar-fill ' + getScoreClass(score);
            } else {
                barEl.style.width = '0%';
                barEl.className = 'component-bar-fill';
            }
        }
    }

    function displayQualityEntities(entities) {
        const entitiesSection = document.getElementById('quality-entities-section');
        const entitiesList = document.getElementById('quality-entities-list');

        if (!entitiesSection || !entitiesList) return;

        if (!entities || entities.length === 0) {
            entitiesSection.classList.add('hidden');
            return;
        }

        entitiesSection.classList.remove('hidden');

        entitiesList.innerHTML = entities.map(entity => {
            const validClass = entity.valid ? 'valid' : 'invalid';
            const typeClass = (entity.type || 'unknown').toLowerCase();
            return `
                <span class="entity-tag ${validClass}">
                    <span class="entity-text">${escapeHtml(entity.text)}</span>
                    <span class="entity-type ${typeClass}">${entity.type || 'unknown'}</span>
                </span>
            `;
        }).join('');
    }

    function displayGrammarIssues(errors) {
        const grammarSection = document.getElementById('quality-grammar-section');
        const grammarList = document.getElementById('quality-grammar-list');

        if (!grammarSection || !grammarList) return;

        if (!errors || errors.length === 0) {
            grammarSection.classList.add('hidden');
            return;
        }

        grammarSection.classList.remove('hidden');

        grammarList.innerHTML = errors.map(error => {
            const suggestions = error.suggestions && error.suggestions.length > 0
                ? `<div class="issue-suggestions">Suggestions: ${error.suggestions.join(', ')}</div>`
                : '';
            return `
                <div class="grammar-issue-item">
                    <div class="issue-message">${escapeHtml(error.message)}</div>
                    <span class="issue-text">${escapeHtml(error.text)}</span>
                    ${suggestions}
                </div>
            `;
        }).join('');
    }

    function displayInvalidEntities(invalidEntities) {
        const invalidSection = document.getElementById('quality-invalid-section');
        const invalidList = document.getElementById('quality-invalid-list');

        if (!invalidSection || !invalidList) return;

        if (!invalidEntities || invalidEntities.length === 0) {
            invalidSection.classList.add('hidden');
            return;
        }

        invalidSection.classList.remove('hidden');

        invalidList.innerHTML = invalidEntities.map(entity => {
            return `
                <div class="invalid-entity-item">
                    <span class="entity-name">${escapeHtml(entity)}</span>
                    <div class="entity-suggestion">This term was not found in the medical lexicon. It may be misspelled.</div>
                </div>
            `;
        }).join('');
    }

    function displayContradictions(contradictions) {
        const contradictionSection = document.getElementById('quality-contradiction-section');
        const contradictionList = document.getElementById('quality-contradiction-list');

        if (!contradictionSection || !contradictionList) return;

        if (!contradictions || contradictions.length === 0) {
            contradictionSection.classList.add('hidden');
            return;
        }

        contradictionSection.classList.remove('hidden');

        contradictionList.innerHTML = contradictions.map(c => {
            const severityClass = c.severity === 'critical' ? 'critical' : (c.severity === 'high' ? 'high' : 'medium');
            return `
                <div class="contradiction-item ${severityClass}">
                    <div class="contradiction-header">
                        <span class="contradiction-type">${c.type.replace('_', ' ')}</span>
                        <span class="contradiction-severity ${severityClass}">${c.severity}</span>
                    </div>
                    <div class="contradiction-statements">
                        <div class="statement-1">"${escapeHtml(c.statement1.substring(0, 100))}${c.statement1.length > 100 ? '...' : ''}"</div>
                        <div class="vs">vs</div>
                        <div class="statement-2">"${escapeHtml(c.statement2.substring(0, 100))}${c.statement2.length > 100 ? '...' : ''}"</div>
                    </div>
                    <div class="contradiction-entity">Entity: <strong>${escapeHtml(c.entity)}</strong></div>
                    <div class="contradiction-explanation">${escapeHtml(c.explanation)}</div>
                </div>
            `;
        }).join('');
    }

    function displayDriftPoints(driftPoints) {
        const driftSection = document.getElementById('quality-drift-section');
        const driftList = document.getElementById('quality-drift-list');

        if (!driftSection || !driftList) return;

        if (!driftPoints || driftPoints.length === 0) {
            driftSection.classList.add('hidden');
            return;
        }

        driftSection.classList.remove('hidden');

        driftList.innerHTML = driftPoints.map(dp => {
            const anomalyClass = dp.is_anomaly ? 'anomaly' : '';
            return `
                <div class="drift-item ${anomalyClass}">
                    <div class="drift-header">
                        <span class="drift-segment">Segment ${dp.segment_index}</span>
                        <span class="drift-similarity">${(dp.similarity * 100).toFixed(1)}% similarity</span>
                        ${dp.is_anomaly ? '<span class="drift-anomaly-badge">Anomaly</span>' : ''}
                    </div>
                    <div class="drift-segments">
                        <div class="drift-from">From: "${escapeHtml(dp.from_segment)}"</div>
                        <div class="drift-to">To: "${escapeHtml(dp.to_segment)}"</div>
                    </div>
                    ${dp.drop_magnitude > 0 ? `<div class="drift-magnitude">Similarity drop: ${(dp.drop_magnitude * 100).toFixed(1)}%</div>` : ''}
                </div>
            `;
        }).join('');
    }

    function displayConfidenceDrops(dropPoints) {
        const confidenceSection = document.getElementById('quality-confidence-section');
        const confidenceList = document.getElementById('quality-confidence-list');

        if (!confidenceSection || !confidenceList) return;

        if (!dropPoints || dropPoints.length === 0) {
            confidenceSection.classList.add('hidden');
            return;
        }

        confidenceSection.classList.remove('hidden');

        confidenceList.innerHTML = dropPoints.map(dp => {
            const anomalyClass = dp.is_anomaly ? 'anomaly' : '';
            return `
                <div class="confidence-item ${anomalyClass}">
                    <div class="confidence-header">
                        <span class="confidence-token">"${escapeHtml(dp.token)}"</span>
                        <span class="confidence-prob">Log prob: ${dp.log_prob.toFixed(2)}</span>
                        ${dp.is_anomaly ? '<span class="confidence-anomaly-badge">Anomaly</span>' : ''}
                    </div>
                    <div class="confidence-context">Context: "${escapeHtml(dp.context)}"</div>
                    ${dp.drop_magnitude > 0 ? `<div class="confidence-magnitude">Probability drop: ${dp.drop_magnitude.toFixed(2)}</div>` : ''}
                </div>
            `;
        }).join('');
    }

    function displayMultiNLPResults(results) {
        if (resultsPlaceholder) {
            resultsPlaceholder.classList.add('hidden');
        }
        resultsSection.classList.remove('hidden');

        // Show multi-NLP section
        const multiNlpSection = document.getElementById('multi-nlp-section');
        multiNlpSection.classList.remove('hidden');

        // Update overall score with best model
        const overallScoreValue = document.getElementById('overall-score-value');
        const overallScoreCircle = document.getElementById('overall-score-circle');

        if (results.best_model && results.models[results.best_model]) {
            const bestScore = results.models[results.best_model].f1_score;
            overallScoreValue.textContent = formatPercent(bestScore);
            overallScoreCircle.className = 'score-circle ' + getScoreClass(bestScore);
        }

        // Build comparison table
        const models = results.models;
        const modelNames = Object.keys(models).sort();

        let html = `
            <table class="backend-comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>GT Entities</th>
                        <th>Pred Entities</th>
                        <th>Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
        `;

        for (const name of modelNames) {
            const m = models[name];
            const isBest = name === results.best_model;
            html += `
                <tr class="${isBest ? 'best-backend' : ''}">
                    <td><strong>${name}</strong>${isBest ? ' ⭐' : ''}</td>
                    <td class="${getScoreClass(m.precision)}">${formatPercent(m.precision)}</td>
                    <td class="${getScoreClass(m.recall)}">${formatPercent(m.recall)}</td>
                    <td class="${getScoreClass(m.f1_score)}">${formatPercent(m.f1_score)}</td>
                    <td>${m.gt_entity_count}</td>
                    <td>${m.pred_entity_count}</td>
                    <td>${m.extraction_time_ms.toFixed(1)}</td>
                </tr>
            `;
        }

        html += `
                </tbody>
            </table>
            <div class="backend-summary">
                <p><strong>Best Model:</strong> ${results.best_model}</p>
                <p><strong>Agreement Rate:</strong> ${formatPercent(results.agreement_rate)}</p>
            </div>
        `;

        document.getElementById('multi-nlp-results').innerHTML = html;

        // Show radar chart
        displayRadarChart(results);

        // Scroll to results on mobile
        if (window.innerWidth <= 1200) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function displayMultiBackendResults(results) {
        if (resultsPlaceholder) {
            resultsPlaceholder.classList.add('hidden');
        }
        resultsSection.classList.remove('hidden');

        // Hide single-backend TER card
        document.getElementById('ter-card')?.classList.add('hidden');

        // Show/create multi-backend section
        let multiBackendSection = document.getElementById('multi-backend-section');
        if (!multiBackendSection) {
            multiBackendSection = document.createElement('div');
            multiBackendSection.id = 'multi-backend-section';
            multiBackendSection.className = 'metric-card';
            const metricsGrid = document.querySelector('.metrics-grid');
            metricsGrid.insertBefore(multiBackendSection, metricsGrid.firstChild);
        }
        multiBackendSection.classList.remove('hidden');

        // Build comparison table
        const backends = results.backends;
        const backendNames = Object.keys(backends).sort();

        let tableHTML = `
            <h3>Multi-Backend TER Comparison</h3>
            <p class="metric-desc">Comparing term extraction across different NLP backends</p>
            <table class="backend-comparison-table">
                <thead>
                    <tr>
                        <th>Backend</th>
                        <th>TER</th>
                        <th>Accuracy</th>
                        <th>GT Terms</th>
                        <th>Pred Terms</th>
                        <th>Correct</th>
                        <th>Sub</th>
                        <th>Del</th>
                        <th>Ins</th>
                    </tr>
                </thead>
                <tbody>
        `;

        for (const name of backendNames) {
            const b = backends[name];
            const isBest = name === results.best_backend;
            tableHTML += `
                <tr class="${isBest ? 'best-backend' : ''}">
                    <td><strong>${name}</strong>${isBest ? ' ⭐' : ''}</td>
                    <td class="${getScoreClass(1 - b.ter)}">${formatPercent(b.ter)}</td>
                    <td class="${getScoreClass(b.accuracy)}">${formatPercent(b.accuracy)}</td>
                    <td>${b.gt_terms}</td>
                    <td>${b.pred_terms}</td>
                    <td>${b.correct}</td>
                    <td>${b.substitutions}</td>
                    <td>${b.deletions}</td>
                    <td>${b.insertions}</td>
                </tr>
            `;
        }

        tableHTML += `
                </tbody>
            </table>
            <div class="backend-summary">
                <p><strong>Best Backend:</strong> ${results.best_backend || 'N/A'}</p>
                <p><strong>Average TER:</strong> ${formatPercent(results.average_ter)}</p>
            </div>
        `;

        multiBackendSection.innerHTML = tableHTML;

        // Update overall score
        const overallScoreValue = document.getElementById('overall-score-value');
        const overallScoreCircle = document.getElementById('overall-score-circle');
        if (results.best_backend && backends[results.best_backend]) {
            const score = backends[results.best_backend].accuracy;
            overallScoreValue.textContent = formatPercent(score);
            overallScoreCircle.className = 'score-circle ' + getScoreClass(score);
        }

        // Scroll to results on mobile
        if (window.innerWidth <= 1200) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function displayErrors(results) {
        const errorsSection = document.getElementById('errors-section');
        const errorsList = document.getElementById('errors-list');

        if (!errorsSection || !errorsList) return;

        if (!results.ter || !results.ter.errors || results.ter.errors.length === 0) {
            errorsSection.classList.add('hidden');
            return;
        }

        errorsSection.classList.remove('hidden');

        const errors = results.ter.errors;
        errorsList.innerHTML = errors.map(error => {
            let severity = 'medium';
            let details = '';

            if (error.type === 'substitution') {
                if (error.category === 'drug') {
                    severity = 'critical';
                    details = `<code>${error.ground_truth}</code> → <code>${error.predicted}</code> (Drug name changed!)`;
                } else if (error.category === 'dosage') {
                    severity = 'high';
                    details = `<code>${error.ground_truth}</code> → <code>${error.predicted}</code> (Dosage changed!)`;
                } else {
                    details = `<code>${error.ground_truth}</code> → <code>${error.predicted}</code>`;
                }
            } else if (error.type === 'deletion') {
                severity = error.category === 'drug' ? 'critical' : 'high';
                details = `<code>${error.ground_truth}</code> (omitted)`;
            } else if (error.type === 'insertion') {
                details = `<code>${error.predicted}</code> (extra)`;
            }

            return `
                <div class="error-item ${severity}">
                    <div class="error-type">${error.type} ${error.category ? `(${error.category})` : ''}</div>
                    <div class="error-detail">${details}</div>
                </div>
            `;
        }).join('');
    }

    // ========================================================================
    // Diff View
    // ========================================================================

    function displayDiff(groundTruth, predicted) {
        const diffGt = document.getElementById('diff-gt');
        const diffPred = document.getElementById('diff-pred');

        // Word-level diff using LCS (Longest Common Subsequence) approach
        const gtWords = groundTruth.split(/\s+/).filter(w => w.length > 0);
        const predWords = predicted.split(/\s+/).filter(w => w.length > 0);

        const diff = computeWordDiff(gtWords, predWords);

        diffGt.innerHTML = diff.gtHtml;
        diffPred.innerHTML = diff.predHtml;
    }

    function computeWordDiff(gtWords, predWords) {
        // Build LCS table
        const m = gtWords.length;
        const n = predWords.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (gtWords[i-1].toLowerCase() === predWords[j-1].toLowerCase()) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        // Backtrack to find the diff
        const gtResult = [];
        const predResult = [];
        let i = m, j = n;

        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && gtWords[i-1].toLowerCase() === predWords[j-1].toLowerCase()) {
                // Words match
                gtResult.unshift({ word: gtWords[i-1], type: 'same' });
                predResult.unshift({ word: predWords[j-1], type: 'same' });
                i--; j--;
            } else if (i > 0 && j > 0 && dp[i-1][j-1] >= dp[i-1][j] && dp[i-1][j-1] >= dp[i][j-1]) {
                // Substitution - words are different at same position
                gtResult.unshift({ word: gtWords[i-1], type: 'deleted', substitute: predWords[j-1] });
                predResult.unshift({ word: predWords[j-1], type: 'inserted', substitute: gtWords[i-1] });
                i--; j--;
            } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
                // Insertion in predicted
                predResult.unshift({ word: predWords[j-1], type: 'inserted' });
                j--;
            } else {
                // Deletion from ground truth
                gtResult.unshift({ word: gtWords[i-1], type: 'deleted' });
                i--;
            }
        }

        // Render HTML
        const gtHtml = gtResult.map(item => {
            if (item.type === 'deleted') {
                const title = item.substitute ? `Changed to: "${item.substitute}"` : 'Deleted';
                return `<span class="diff-deleted" title="${title}">${escapeHtml(item.word)}</span>`;
            }
            return escapeHtml(item.word);
        }).join(' ');

        const predHtml = predResult.map(item => {
            if (item.type === 'inserted') {
                const title = item.substitute ? `Was: "${item.substitute}"` : 'Inserted';
                return `<span class="diff-inserted" title="${title}">${escapeHtml(item.word)}</span>`;
            }
            return escapeHtml(item.word);
        }).join(' ');

        return { gtHtml, predHtml };
    }

    // ========================================================================
    // Radar Chart
    // ========================================================================

    function displayRadarChart(results) {
        const chartContainer = document.getElementById('chart-container');
        const canvas = document.getElementById('metrics-radar-chart');

        if (!results.models || Object.keys(results.models).length === 0) {
            chartContainer.classList.add('hidden');
            return;
        }

        chartContainer.classList.remove('hidden');

        // Prepare data
        const labels = ['Precision', 'Recall', 'F1 Score'];
        const datasets = [];
        const colors = [
            'rgba(37, 99, 235, 0.7)',   // Blue
            'rgba(34, 197, 94, 0.7)',   // Green
            'rgba(249, 115, 22, 0.7)',  // Orange
            'rgba(168, 85, 247, 0.7)',  // Purple
            'rgba(236, 72, 153, 0.7)',  // Pink
        ];

        let colorIndex = 0;
        for (const [name, metrics] of Object.entries(results.models)) {
            datasets.push({
                label: name,
                data: [metrics.precision, metrics.recall, metrics.f1_score],
                backgroundColor: colors[colorIndex % colors.length].replace('0.7', '0.2'),
                borderColor: colors[colorIndex % colors.length],
                borderWidth: 2,
                pointBackgroundColor: colors[colorIndex % colors.length],
            });
            colorIndex++;
        }

        // Destroy previous chart
        if (radarChart) {
            radarChart.destroy();
        }

        // Create new chart
        radarChart = new Chart(canvas, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            stepSize: 0.2,
                        },
                    },
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                },
            },
        });
    }

    // ========================================================================
    // Export Functions
    // ========================================================================

    function exportJSON() {
        if (!lastResults) {
            alert('No results to export. Run an evaluation first.');
            return;
        }

        const blob = new Blob([JSON.stringify(lastResults, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'hsttb-results.json');
    }

    function exportCSV() {
        if (!lastResults) {
            alert('No results to export. Run an evaluation first.');
            return;
        }

        let csv = 'Metric,Value\n';

        // Evaluation mode
        csv += `Mode,${lastResults.mode || 'unknown'}\n`;

        if (lastResults.wer) {
            csv += `WER,${lastResults.wer.wer}\n`;
            csv += `WER Word Accuracy,${lastResults.wer.word_accuracy}\n`;
            csv += `WER Substitutions,${lastResults.wer.substitutions}\n`;
            csv += `WER Deletions,${lastResults.wer.deletions}\n`;
            csv += `WER Insertions,${lastResults.wer.insertions}\n`;
        }

        if (lastResults.cer) {
            csv += `CER,${lastResults.cer.cer}\n`;
            csv += `CER Char Accuracy,${lastResults.cer.char_accuracy}\n`;
            csv += `CER Substitutions,${lastResults.cer.substitutions}\n`;
            csv += `CER Deletions,${lastResults.cer.deletions}\n`;
            csv += `CER Insertions,${lastResults.cer.insertions}\n`;
        }

        if (lastResults.ter) {
            csv += `TER Overall,${lastResults.ter.overall_ter}\n`;
            csv += `TER Substitutions,${lastResults.ter.substitutions}\n`;
            csv += `TER Deletions,${lastResults.ter.deletions}\n`;
            csv += `TER Insertions,${lastResults.ter.insertions}\n`;
        }

        if (lastResults.ner) {
            csv += `NER Precision,${lastResults.ner.precision}\n`;
            csv += `NER Recall,${lastResults.ner.recall}\n`;
            csv += `NER F1,${lastResults.ner.f1_score}\n`;
        }

        if (lastResults.crs) {
            csv += `CRS Composite,${lastResults.crs.composite_score}\n`;
            csv += `CRS Semantic Similarity,${lastResults.crs.semantic_similarity}\n`;
            csv += `CRS Entity Continuity,${lastResults.crs.entity_continuity}\n`;
        }

        if (lastResults.quality) {
            csv += `Quality Composite,${lastResults.quality.composite_score}\n`;
            csv += `Quality Perplexity,${lastResults.quality.perplexity}\n`;
            csv += `Quality Perplexity Score,${lastResults.quality.perplexity_score}\n`;
            csv += `Quality Grammar Score,${lastResults.quality.grammar_score}\n`;
            csv += `Quality Entity Validity,${lastResults.quality.entity_validity_score}\n`;
            csv += `Quality Coherence,${lastResults.quality.coherence_score}\n`;
            csv += `Quality Contradiction Score,${lastResults.quality.contradiction_score}\n`;
            csv += `Quality Semantic Stability,${lastResults.quality.embedding_drift_score}\n`;
            csv += `Quality Confidence Score,${lastResults.quality.confidence_variance_score}\n`;
            csv += `Quality Recommendation,${lastResults.quality.recommendation}\n`;
            csv += `Quality Word Count,${lastResults.quality.word_count}\n`;
            csv += `Quality Medical Entity Count,${lastResults.quality.medical_entity_count}\n`;
            csv += `Quality Contradictions Found,${lastResults.quality.contradictions ? lastResults.quality.contradictions.length : 0}\n`;
            csv += `Quality Drift Points,${lastResults.quality.drift_points ? lastResults.quality.drift_points.length : 0}\n`;
            csv += `Quality Confidence Drops,${lastResults.quality.confidence_drop_points ? lastResults.quality.confidence_drop_points.length : 0}\n`;
        }

        if (lastResults.overall_score !== undefined) {
            csv += `Overall Score,${lastResults.overall_score}\n`;
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        downloadBlob(blob, 'hsttb-results.csv');
    }

    function downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ========================================================================
    // Examples and Backends
    // ========================================================================

    async function loadExamples() {
        try {
            const response = await fetch('/api/examples');
            const data = await response.json();

            examplesContainer.innerHTML = data.examples.map((ex, i) => `
                <div class="example-card" data-index="${i}">
                    <div class="example-name">${ex.name}</div>
                    <div class="example-desc">${ex.description}</div>
                </div>
            `).join('');

            window.examplesData = data.examples;

            document.querySelectorAll('.example-card').forEach(card => {
                card.addEventListener('click', () => {
                    const index = parseInt(card.dataset.index);
                    const example = window.examplesData[index];
                    groundTruthInput.value = example.ground_truth;
                    predictedInput.value = example.predicted;
                    groundTruthInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                });
            });
        } catch (error) {
            console.error('Failed to load examples:', error);
            examplesContainer.innerHTML = '<p style="color: var(--text-muted);">Failed to load examples</p>';
        }
    }

    async function loadBackends() {
        try {
            const response = await fetch('/api/backends');
            const data = await response.json();
            window.availableBackends = data;
            console.log('Available NLP backends:', data.backends);
        } catch (error) {
            console.error('Failed to load backends:', error);
        }
    }

    async function loadNLPModels() {
        try {
            const response = await fetch('/api/nlp-models');
            const data = await response.json();

            if (data.status === 'success') {
                const availableModels = [];
                const unavailableModels = [];

                data.models.forEach(model => {
                    const statusEl = document.getElementById(`status-${model.name}`);
                    const checkbox = document.querySelector(`input[value="${model.name}"]`);

                    if (model.available) {
                        availableModels.push(model.name);
                        if (statusEl) {
                            statusEl.textContent = '✓';
                            statusEl.className = 'model-status available';
                        }
                    } else {
                        unavailableModels.push({ name: model.name, error: model.error });
                        if (statusEl) {
                            statusEl.textContent = '✗';
                            statusEl.className = 'model-status unavailable';
                            statusEl.title = model.error || 'Not installed';
                        }
                        if (checkbox) {
                            checkbox.disabled = true;
                            checkbox.checked = false;
                            checkbox.parentElement.classList.add('disabled');
                            checkbox.parentElement.title = model.error || 'Not installed';
                        }
                    }
                });

                // Update note
                const noteEl = document.getElementById('model-availability-note');
                if (noteEl && unavailableModels.length > 0) {
                    const names = unavailableModels.map(m => m.name).join(', ');
                    noteEl.innerHTML = `<small>⚠️ Some models not installed: ${names}. Install with <code>pip install medspacy</code></small>`;
                }

                console.log('Available NLP models:', availableModels);
                if (unavailableModels.length > 0) {
                    console.warn('Unavailable NLP models:', unavailableModels);
                }
            }
        } catch (error) {
            console.error('Failed to load NLP models:', error);
        }
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    function clearForm() {
        groundTruthInput.value = '';
        predictedInput.value = '';
        resultsSection.classList.add('hidden');
        if (resultsPlaceholder) {
            resultsPlaceholder.classList.remove('hidden');
        }
        // Hide speech rate section
        const speechRateSection = document.getElementById('speech-rate-section');
        if (speechRateSection) {
            speechRateSection.classList.add('hidden');
        }
        removeAudio();
        groundTruthInput.focus();
    }

    function formatPercent(value) {
        return (value * 100).toFixed(1) + '%';
    }

    function formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function getScoreClass(score) {
        if (score >= 0.9) return 'score-good';
        if (score >= 0.7) return 'score-warning';
        return 'score-bad';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
