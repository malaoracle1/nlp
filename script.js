// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const elements = {
    // Status elements
    apiStatus: document.getElementById('api-status'),
    modelStatus: document.getElementById('model-status'),
    modelText: document.getElementById('model-text'),
    refreshStatusBtn: document.getElementById('refresh-status'),

    // Training elements
    useAugmentationCheckbox: document.getElementById('use-augmentation'),
    trainModelBtn: document.getElementById('train-model'),
    trainingResults: document.getElementById('training-results'),
    metricsDisplay: document.getElementById('metrics-display'),

    // Single prediction elements
    commentInput: document.getElementById('comment-input'),
    charCount: document.getElementById('char-count'),
    predictSingleBtn: document.getElementById('predict-single'),
    singleResult: document.getElementById('single-result'),
    resultLabel: document.getElementById('result-label'),
    confidenceBadge: document.getElementById('confidence-badge'),
    probabilityFill: document.getElementById('probability-fill'),
    probabilityText: document.getElementById('probability-text'),

    // Batch prediction elements
    batchInput: document.getElementById('batch-input'),
    predictBatchBtn: document.getElementById('predict-batch'),
    batchResults: document.getElementById('batch-results'),
    batchResultsContainer: document.getElementById('batch-results-container'),

    // Sample elements
    sampleItems: document.querySelectorAll('.sample-item'),

    // Toast
    toast: document.getElementById('toast')
};

// Utility Functions
function showToast(message, type = 'info') {
    const toast = elements.toast;
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function setButtonLoading(button, isLoading, loadingText = 'Loading...') {
    const textSpan = button.querySelector('.btn-text');
    const loaderSpan = button.querySelector('.btn-loader');

    if (isLoading) {
        button.disabled = true;
        textSpan.style.display = 'none';
        loaderSpan.style.display = 'inline';
        loaderSpan.textContent = loadingText;
    } else {
        button.disabled = false;
        textSpan.style.display = 'inline';
        loaderSpan.style.display = 'none';
    }
}

function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

function updateStatusIndicator(isOnline, statusText) {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusTextSpan = elements.apiStatus.querySelector('.status-text');

    statusDot.className = `status-dot ${isOnline ? 'online' : 'offline'}`;
    statusTextSpan.textContent = statusText;
}

// API Functions
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            updateStatusIndicator(true, 'API Online');
            return true;
        } else {
            updateStatusIndicator(false, 'API Error');
            return false;
        }
    } catch (error) {
        updateStatusIndicator(false, 'API Offline');
        console.error('API Status Error:', error);
        return false;
    }
}

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/status`);
        if (response.ok) {
            const data = await response.json();
            const statusText = data.model_loaded ? 'Model Ready' : 'Model Not Trained';
            elements.modelText.textContent = statusText;
            elements.modelText.style.color = data.model_loaded ? '#28a745' : '#ffc107';
            return data;
        } else {
            elements.modelText.textContent = 'Status Unknown';
            elements.modelText.style.color = '#dc3545';
            return null;
        }
    } catch (error) {
        elements.modelText.textContent = 'Status Error';
        elements.modelText.style.color = '#dc3545';
        console.error('Model Status Error:', error);
        return null;
    }
}

async function trainModel(useAugmentation = true) {
    try {
        setButtonLoading(elements.trainModelBtn, true, 'Training...');
        showToast('Starting model training...', 'info');

        const response = await fetch(`${API_BASE_URL}/model/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                use_augmentation: useAugmentation
            })
        });

        const data = await response.json();

        if (response.ok) {
            showToast('Model trained successfully!', 'success');
            displayTrainingResults(data.metrics);
            await checkModelStatus(); // Refresh model status
        } else {
            throw new Error(data.detail || 'Training failed');
        }
    } catch (error) {
        console.error('Training Error:', error);
        showToast(`Training failed: ${error.message}`, 'error');
    } finally {
        setButtonLoading(elements.trainModelBtn, false);
    }
}

async function predictSingle(text) {
    try {
        setButtonLoading(elements.predictSingleBtn, true, 'Analyzing...');

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });

        const data = await response.json();

        if (response.ok) {
            displaySingleResult(data);
        } else {
            throw new Error(data.detail || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction Error:', error);
        showToast(`Prediction failed: ${error.message}`, 'error');
    } finally {
        setButtonLoading(elements.predictSingleBtn, false);
    }
}

async function predictBatch(texts) {
    try {
        setButtonLoading(elements.predictBatchBtn, true, 'Analyzing...');

        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(texts)
        });

        const data = await response.json();

        if (response.ok) {
            displayBatchResults(data.predictions);
        } else {
            throw new Error(data.detail || 'Batch prediction failed');
        }
    } catch (error) {
        console.error('Batch Prediction Error:', error);
        showToast(`Batch prediction failed: ${error.message}`, 'error');
    } finally {
        setButtonLoading(elements.predictBatchBtn, false);
    }
}

// Display Functions
function displayTrainingResults(metrics) {
    elements.trainingResults.style.display = 'block';

    const metricsHtml = `
        <div class="metrics-grid">
            <div class="metric-item">
                <span class="metric-value">${formatPercentage(metrics.accuracy)}</span>
                <span class="metric-label">Accuracy</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${formatPercentage(metrics.f1_score)}</span>
                <span class="metric-label">F1-Score</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${formatPercentage(metrics.roc_auc)}</span>
                <span class="metric-label">ROC-AUC</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${metrics.true_positives}</span>
                <span class="metric-label">True Positives</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${metrics.true_negatives}</span>
                <span class="metric-label">True Negatives</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${metrics.false_positives}</span>
                <span class="metric-label">False Positives</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${metrics.false_negatives}</span>
                <span class="metric-label">False Negatives</span>
            </div>
            <div class="metric-item">
                <span class="metric-value">${metrics.total_samples}</span>
                <span class="metric-label">Test Samples</span>
            </div>
        </div>
    `;

    elements.metricsDisplay.innerHTML = metricsHtml;
}

function displaySingleResult(result) {
    const isToxic = result.prediction === 1;

    // Update result label
    elements.resultLabel.textContent = isToxic ? '⚠️ Toxic Comment' : '✅ Safe Comment';
    elements.resultLabel.className = `result-label ${isToxic ? 'toxic' : 'safe'}`;

    // Update confidence badge
    elements.confidenceBadge.textContent = result.confidence;
    elements.confidenceBadge.className = `confidence-badge ${result.confidence}`;

    // Update probability bar
    const probabilityPercent = (result.probability * 100);
    elements.probabilityFill.style.width = `${probabilityPercent}%`;
    elements.probabilityText.textContent = `${probabilityPercent.toFixed(1)}% toxic`;

    // Show result
    elements.singleResult.style.display = 'block';

    // Update result card border color
    elements.singleResult.style.borderLeftColor = isToxic ? '#dc3545' : '#28a745';
    elements.singleResult.style.background = isToxic ? '#fdebee' : '#e8f5e8';
}

function displayBatchResults(predictions) {
    let resultsHtml = '';

    predictions.forEach((prediction, index) => {
        const isToxic = prediction.prediction === 1;
        const resultClass = isToxic ? 'toxic' : 'safe';
        const resultIcon = isToxic ? '⚠️' : '✅';
        const resultText = isToxic ? 'Toxic' : 'Safe';

        resultsHtml += `
            <div class="batch-item ${resultClass}">
                <div class="batch-text">"${prediction.text}"</div>
                <div class="batch-meta">
                    <span><strong>${resultIcon} ${resultText}</strong></span>
                    <span>Probability: ${formatPercentage(prediction.probability)} | Confidence: ${prediction.confidence}</span>
                </div>
            </div>
        `;
    });

    elements.batchResultsContainer.innerHTML = resultsHtml;
    elements.batchResults.style.display = 'block';
}

// Event Listeners
function initializeEventListeners() {
    // Character counter for comment input
    elements.commentInput.addEventListener('input', function() {
        const length = this.value.length;
        elements.charCount.textContent = length;

        if (length > 900) {
            elements.charCount.style.color = '#dc3545';
        } else if (length > 700) {
            elements.charCount.style.color = '#ffc107';
        } else {
            elements.charCount.style.color = '#6c757d';
        }
    });

    // Refresh status button
    elements.refreshStatusBtn.addEventListener('click', async function() {
        setButtonLoading(this, true, 'Checking...');
        await Promise.all([checkApiStatus(), checkModelStatus()]);
        setButtonLoading(this, false);
        showToast('Status refreshed', 'success');
    });

    // Train model button
    elements.trainModelBtn.addEventListener('click', function() {
        const useAugmentation = elements.useAugmentationCheckbox.checked;
        trainModel(useAugmentation);
    });

    // Single prediction button
    elements.predictSingleBtn.addEventListener('click', function() {
        const text = elements.commentInput.value.trim();
        if (!text) {
            showToast('Please enter a comment to analyze', 'error');
            return;
        }
        predictSingle(text);
    });

    // Batch prediction button
    elements.predictBatchBtn.addEventListener('click', function() {
        const text = elements.batchInput.value.trim();
        if (!text) {
            showToast('Please enter comments to analyze', 'error');
            return;
        }

        const texts = text.split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0);

        if (texts.length === 0) {
            showToast('Please enter at least one comment', 'error');
            return;
        }

        predictBatch(texts);
    });

    // Sample comment items
    elements.sampleItems.forEach(item => {
        item.addEventListener('click', function() {
            const sampleText = this.getAttribute('data-text');
            elements.commentInput.value = sampleText;
            elements.commentInput.dispatchEvent(new Event('input')); // Trigger character counter

            // Smooth scroll to prediction section
            elements.commentInput.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Optional: Auto-predict after a short delay
            setTimeout(() => {
                if (elements.commentInput.value === sampleText) {
                    elements.predictSingleBtn.click();
                }
            }, 500);
        });
    });

    // Enter key shortcuts
    elements.commentInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            elements.predictSingleBtn.click();
        }
    });

    elements.batchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            elements.predictBatchBtn.click();
        }
    });
}

// Initialize Application
async function initializeApp() {
    console.log('Initializing Toxic Comment Classifier...');

    // Initialize event listeners
    initializeEventListeners();

    // Check initial status
    showToast('Checking API status...', 'info');
    await Promise.all([checkApiStatus(), checkModelStatus()]);

    console.log('Application initialized successfully');
}

// Start the application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Add some helpful keyboard shortcuts info
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter in textareas triggers prediction
    if (e.key === 'Enter' && e.ctrlKey) {
        if (document.activeElement === elements.commentInput) {
            elements.predictSingleBtn.click();
        } else if (document.activeElement === elements.batchInput) {
            elements.predictBatchBtn.click();
        }
    }
});

// Auto-refresh status every 30 seconds
setInterval(async () => {
    await checkApiStatus();
}, 30000);

// Export for debugging (if needed)
window.ToxicClassifierApp = {
    checkApiStatus,
    checkModelStatus,
    trainModel,
    predictSingle,
    predictBatch,
    elements
};