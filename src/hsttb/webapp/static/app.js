// HSTTB Web Application JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
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

    // Load examples
    loadExamples();

    // Event Listeners
    evaluateBtn.addEventListener('click', runEvaluation);
    clearBtn.addEventListener('click', clearForm);

    // Enter key to submit
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            runEvaluation();
        }
    });

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

            // Store examples for click handling
            window.examplesData = data.examples;

            // Add click handlers
            document.querySelectorAll('.example-card').forEach(card => {
                card.addEventListener('click', () => {
                    const index = parseInt(card.dataset.index);
                    const example = window.examplesData[index];
                    groundTruthInput.value = example.ground_truth;
                    predictedInput.value = example.predicted;
                    // Scroll to form
                    groundTruthInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                });
            });
        } catch (error) {
            console.error('Failed to load examples:', error);
            examplesContainer.innerHTML = '<p style="color: var(--text-muted);">Failed to load examples</p>';
        }
    }

    async function runEvaluation() {
        const groundTruth = groundTruthInput.value.trim();
        const predicted = predictedInput.value.trim();

        if (!groundTruth || !predicted) {
            alert('Please enter both ground truth and predicted text.');
            return;
        }

        // Disable button and show loading
        evaluateBtn.disabled = true;
        evaluateBtn.textContent = 'Evaluating...';
        evaluateBtn.classList.add('loading');

        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
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
                displayResults(results);
            } else {
                alert('Evaluation failed: ' + (results.error || 'Unknown error'));
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

    function displayResults(results) {
        resultsSection.classList.remove('hidden');

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
        const terCard = document.getElementById('ter-card');
        const terValue = document.getElementById('ter-value');
        const terDetails = document.getElementById('ter-details');

        if (results.ter) {
            terCard.classList.remove('hidden');
            // For TER, lower is better, so invert for color coding
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

        // NER Results
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

        // CRS Results
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

        // Display errors
        displayErrors(results);

        // Scroll to results
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
                    details = `<code>${error.ground_truth}</code> &rarr; <code>${error.predicted}</code> (Drug name changed!)`;
                } else if (error.category === 'dosage') {
                    severity = 'high';
                    details = `<code>${error.ground_truth}</code> &rarr; <code>${error.predicted}</code> (Dosage changed!)`;
                } else {
                    details = `<code>${error.ground_truth}</code> &rarr; <code>${error.predicted}</code>`;
                }
            } else if (error.type === 'deletion') {
                if (error.category === 'drug') {
                    severity = 'critical';
                    details = `<code>${error.ground_truth}</code> (Drug name omitted!)`;
                } else {
                    severity = 'high';
                    details = `<code>${error.ground_truth}</code> (Term omitted)`;
                }
            } else if (error.type === 'insertion') {
                details = `<code>${error.predicted}</code> (Extra term inserted)`;
            }

            return `
                <div class="error-item ${severity}">
                    <div class="error-type">${error.type} ${error.category ? `(${error.category})` : ''}</div>
                    <div class="error-detail">${details}</div>
                </div>
            `;
        }).join('');
    }

    function clearForm() {
        groundTruthInput.value = '';
        predictedInput.value = '';
        resultsSection.classList.add('hidden');
        groundTruthInput.focus();
    }

    function formatPercent(value) {
        return (value * 100).toFixed(1) + '%';
    }

    function getScoreClass(score) {
        if (score >= 0.9) return 'score-good';
        if (score >= 0.7) return 'score-warning';
        return 'score-bad';
    }
});
