/**
 * Customer Service Support System - Representative View
 */

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('query-form');
    const questionInput = document.getElementById('question');
    const submitBtn = document.getElementById('submit-btn');
    const loadingSection = document.getElementById('loading');
    const answerSection = document.getElementById('answer-section');

    // Answer elements
    const answerContent = document.getElementById('answer-content');
    const confidenceBadge = document.getElementById('confidence-badge');
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceLabel = document.getElementById('confidence-label');
    const sourceLink = document.getElementById('source-link');
    const intentBadge = document.getElementById('intent-badge');
    const responseTime = document.getElementById('response-time');
    const reformulatedQuery = document.getElementById('reformulated-query');

    // Modal elements
    const modal = document.getElementById('document-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const modalClose = document.getElementById('modal-close');

    // Current source document name
    let currentSourceDoc = '';

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const question = questionInput.value.trim();
        if (!question) return;

        // Show loading, hide answer
        submitBtn.disabled = true;
        loadingSection.classList.remove('hidden');
        answerSection.classList.add('hidden');

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            });

            if (!response.ok) {
                throw new Error('Failed to get answer');
            }

            const data = await response.json();
            displayAnswer(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your question. Please try again.');
        } finally {
            submitBtn.disabled = false;
            loadingSection.classList.add('hidden');
        }
    });

    // Display the answer
    function displayAnswer(data) {
        // Update answer content
        answerContent.textContent = data.answer;

        // Update confidence badge
        confidenceScore.textContent = data.confidence_score + '%';
        confidenceLabel.textContent = getConfidenceLabel(data.confidence_level);

        // Remove all confidence classes and add the appropriate one
        confidenceBadge.classList.remove('high', 'medium', 'low');
        confidenceBadge.classList.add(data.confidence_level);

        // Update metadata
        currentSourceDoc = data.source_document;
        sourceLink.textContent = data.source_document;
        sourceLink.href = '#';

        intentBadge.textContent = data.detected_intent;
        responseTime.textContent = data.response_time_ms + 'ms';
        reformulatedQuery.textContent = data.reformulated_query;

        // Show answer section
        answerSection.classList.remove('hidden');
    }

    function getConfidenceLabel(level) {
        switch (level) {
            case 'high': return 'High Confidence';
            case 'medium': return 'Medium Confidence';
            case 'low': return 'Low Confidence';
            default: return 'Unknown';
        }
    }

    // Source link click - open document modal
    sourceLink.addEventListener('click', async function(e) {
        e.preventDefault();

        if (!currentSourceDoc || currentSourceDoc === 'none') {
            alert('No source document available.');
            return;
        }

        try {
            const response = await fetch(`/api/documents/${encodeURIComponent(currentSourceDoc)}`);

            if (!response.ok) {
                throw new Error('Failed to load document');
            }

            const data = await response.json();
            modalTitle.textContent = data.name;
            modalBody.textContent = data.content;
            modal.classList.remove('hidden');
        } catch (error) {
            console.error('Error loading document:', error);
            alert('Failed to load the source document.');
        }
    });

    // Close modal
    modalClose.addEventListener('click', function() {
        modal.classList.add('hidden');
    });

    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            modal.classList.add('hidden');
        }
    });
});
