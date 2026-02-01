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
    const computeTer = document.getElementById('compute-ter');
    const computeNer = document.getElementById('compute-ner');
    const computeCrs = document.getElementById('compute-crs');
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

    // ========================================================================
    // Initialization
    // ========================================================================

    loadExamples();
    loadBackends();
    loadTTSHistory();
    setupEventListeners();

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

    // ========================================================================
    // Evaluation
    // ========================================================================

    async function runEvaluation() {
        const groundTruth = groundTruthInput.value.trim();
        const predicted = predictedInput.value.trim();

        if (!groundTruth || !predicted) {
            alert('Please enter both ground truth and predicted text.');
            return;
        }

        evaluateBtn.disabled = true;
        evaluateBtn.textContent = 'Evaluating...';
        evaluateBtn.classList.add('loading');

        try {
            if (multiNlpToggle.checked) {
                await runMultiNLPEvaluation(groundTruth, predicted);
            } else if (multiBackendToggle.checked) {
                await runMultiBackendEvaluation(groundTruth, predicted);
            } else {
                await runSingleEvaluation(groundTruth, predicted);
            }
        } catch (error) {
            console.error('Evaluation error:', error);
            alert('Failed to run evaluation. Please try again.');
        } finally {
            evaluateBtn.disabled = false;
            evaluateBtn.textContent = 'Evaluate';
            evaluateBtn.classList.remove('loading');
        }
    }

    async function runSingleEvaluation(groundTruth, predicted) {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ground_truth: groundTruth,
                predicted: predicted,
                compute_ter: computeTer.checked,
                compute_ner: computeNer.checked,
                compute_crs: computeCrs.checked,
            }),
        });

        const results = await response.json();

        if (results.status === 'success') {
            lastResults = results;
            displayResults(results);
            displayDiff(groundTruth, predicted);
        } else {
            alert('Evaluation failed: ' + (results.error || 'Unknown error'));
        }
    }

    async function runMultiBackendEvaluation(groundTruth, predicted) {
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
            lastResults = results;
            displayMultiBackendResults(results);
            displayDiff(groundTruth, predicted);
        } else {
            alert('Evaluation failed: ' + (results.error || 'Unknown error'));
        }
    }

    async function runMultiNLPEvaluation(groundTruth, predicted) {
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
            lastResults = results;
            displayMultiNLPResults(results);
            displayDiff(groundTruth, predicted);
        } else {
            alert('Evaluation failed: ' + (results.error || 'Unknown error'));
        }
    }

    // ========================================================================
    // Results Display
    // ========================================================================

    function displayResults(results) {
        resultsSection.classList.remove('hidden');

        // Hide multi-sections
        document.getElementById('multi-nlp-section')?.classList.add('hidden');
        document.getElementById('multi-backend-section')?.classList.add('hidden');
        document.getElementById('chart-container')?.classList.add('hidden');

        // Overall score
        const overallScoreValue = document.getElementById('overall-score-value');
        const overallScoreCircle = document.getElementById('overall-score-circle');

        if (results.overall_score !== undefined) {
            const score = results.overall_score;
            overallScoreValue.textContent = formatPercent(score);
            overallScoreCircle.className = 'score-circle ' + getScoreClass(score);
        } else {
            overallScoreValue.textContent = '--';
            overallScoreCircle.className = 'score-circle';
        }

        // TER Results
        displayTERResults(results);

        // NER Results
        displayNERResults(results);

        // CRS Results
        displayCRSResults(results);

        // Errors
        displayErrors(results);

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
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

    function displayMultiNLPResults(results) {
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

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function displayMultiBackendResults(results) {
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

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function displayErrors(results) {
        const errorsSection = document.getElementById('errors-section');
        const errorsList = document.getElementById('errors-list');

        if (!results.ter || results.ter.errors.length === 0) {
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

        // Simple word-based diff
        const gtWords = groundTruth.split(/\s+/);
        const predWords = predicted.split(/\s+/);

        // Find differences
        const gtHighlighted = highlightDifferences(gtWords, predWords, 'deleted');
        const predHighlighted = highlightDifferences(predWords, gtWords, 'inserted');

        diffGt.innerHTML = gtHighlighted;
        diffPred.innerHTML = predHighlighted;
    }

    function highlightDifferences(words1, words2, diffClass) {
        const words2Lower = words2.map(w => w.toLowerCase());

        return words1.map(word => {
            if (!words2Lower.includes(word.toLowerCase())) {
                return `<span class="diff-${diffClass}">${escapeHtml(word)}</span>`;
            }
            return escapeHtml(word);
        }).join(' ');
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

    // ========================================================================
    // Utilities
    // ========================================================================

    function clearForm() {
        groundTruthInput.value = '';
        predictedInput.value = '';
        resultsSection.classList.add('hidden');
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
