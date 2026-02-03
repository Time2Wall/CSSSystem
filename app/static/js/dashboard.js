/**
 * Customer Service Support System - Manager Dashboard
 */

// Chart instances
let confidenceChart = null;
let intentChart = null;
let documentsChart = null;
let timelineChart = null;

// Colors
const chartColors = {
    primary: '#2563eb',
    success: '#16a34a',
    warning: '#d97706',
    danger: '#dc2626',
    gray: '#6b7280',
    lightGray: '#e5e7eb'
};

document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();

    // Refresh data every 30 seconds
    setInterval(loadDashboardData, 30000);
});

async function loadDashboardData() {
    try {
        // Load stats and queries in parallel
        const [statsResponse, queriesResponse] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/queries?limit=10')
        ]);

        if (!statsResponse.ok || !queriesResponse.ok) {
            throw new Error('Failed to load data');
        }

        const stats = await statsResponse.json();
        const queries = await queriesResponse.json();

        updateSummaryCards(stats);
        updateCharts(stats);
        updateRecentQueriesTable(queries.queries);
        updateLowConfidenceTable(stats);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateSummaryCards(stats) {
    document.getElementById('total-queries').textContent = stats.total_queries;
    document.getElementById('avg-confidence').textContent = stats.avg_confidence.toFixed(1) + '%';
    document.getElementById('avg-response-time').textContent = stats.avg_response_time_ms.toFixed(0) + 'ms';
    document.getElementById('low-confidence-count').textContent = stats.low_confidence_count;
}

function updateCharts(stats) {
    updateConfidenceChart(stats.confidence_distribution);
    updateIntentChart(stats.intent_distribution);
    updateDocumentsChart(stats.top_documents);
    updateTimelineChart(stats.queries_per_day);
}

function updateConfidenceChart(distribution) {
    const ctx = document.getElementById('confidence-chart').getContext('2d');

    if (confidenceChart) {
        confidenceChart.destroy();
    }

    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['High (70+)', 'Medium (40-69)', 'Low (<40)'],
            datasets: [{
                label: 'Number of Queries',
                data: [distribution.high, distribution.medium, distribution.low],
                backgroundColor: [chartColors.success, chartColors.warning, chartColors.danger],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function updateIntentChart(distribution) {
    const ctx = document.getElementById('intent-chart').getContext('2d');

    if (intentChart) {
        intentChart.destroy();
    }

    const labels = Object.keys(distribution);
    const data = Object.values(distribution);

    intentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#2563eb', '#7c3aed', '#db2777', '#ea580c',
                    '#16a34a', '#0891b2', '#6b7280'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

function updateDocumentsChart(documents) {
    const ctx = document.getElementById('documents-chart').getContext('2d');

    if (documentsChart) {
        documentsChart.destroy();
    }

    const labels = documents.map(d => d.name.replace('.md', ''));
    const data = documents.map(d => d.usage_count);

    documentsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Times Used',
                data: data,
                backgroundColor: chartColors.primary,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function updateTimelineChart(queriesPerDay) {
    const ctx = document.getElementById('timeline-chart').getContext('2d');

    if (timelineChart) {
        timelineChart.destroy();
    }

    // Sort dates and fill in missing days
    const dates = Object.keys(queriesPerDay).sort();
    const data = dates.map(date => queriesPerDay[date]);

    // Format dates for display
    const labels = dates.map(date => {
        const d = new Date(date);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Queries',
                data: data,
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function updateRecentQueriesTable(queries) {
    const tbody = document.getElementById('recent-queries-body');
    const noData = document.getElementById('no-recent-queries');

    if (!queries || queries.length === 0) {
        tbody.innerHTML = '';
        noData.classList.remove('hidden');
        return;
    }

    noData.classList.add('hidden');
    tbody.innerHTML = queries.map(q => `
        <tr>
            <td title="${escapeHtml(q.question)}">${truncate(q.question, 50)}</td>
            <td title="${escapeHtml(q.answer)}">${truncate(q.answer, 40)}</td>
            <td><span class="confidence-badge ${q.confidence_level}">${q.confidence_score}%</span></td>
            <td>${q.source_document}</td>
            <td>${formatTime(q.created_at)}</td>
        </tr>
    `).join('');
}

async function updateLowConfidenceTable(stats) {
    const tbody = document.getElementById('low-confidence-body');
    const noData = document.getElementById('no-low-confidence');

    try {
        const response = await fetch('/api/queries?max_confidence=39&limit=10');
        if (!response.ok) throw new Error('Failed to load low confidence queries');

        const data = await response.json();

        if (!data.queries || data.queries.length === 0) {
            tbody.innerHTML = '';
            noData.classList.remove('hidden');
            return;
        }

        noData.classList.add('hidden');
        tbody.innerHTML = data.queries.map(q => `
            <tr>
                <td title="${escapeHtml(q.question)}">${truncate(q.question, 60)}</td>
                <td><span class="confidence-badge low">${q.confidence_score}%</span></td>
                <td>${q.detected_intent}</td>
                <td>${formatTime(q.created_at)}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading low confidence queries:', error);
        noData.classList.remove('hidden');
    }
}

// Utility functions
function truncate(str, maxLength) {
    if (!str) return '';
    return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}
