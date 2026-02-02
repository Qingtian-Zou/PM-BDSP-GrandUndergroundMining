/**
 * Grand Underground Mining Game - AI Assistance Module
 */

// ===========================================
// AI Hint System
// ===========================================

let currentHintTimeout = null;

async function getAIHint() {
    if (GameState.isGameOver || !GameState.gameId) return;

    const hintBtn = document.getElementById('hint-btn');
    hintBtn.disabled = true;
    hintBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Thinking...</span>';

    try {
        const data = await API.getSuggestion(GameState.gameId);

        if (data.suggestion) {
            showHint(data.suggestion);
        }
    } catch (error) {
        console.error('Failed to get AI suggestion:', error);
    } finally {
        hintBtn.disabled = false;
        hintBtn.innerHTML = '<span class="btn-icon">🤖</span><span>Get AI Hint</span>';
    }
}

function showHint(suggestion) {
    const overlay = document.getElementById('normal-hint-overlay');
    const board = document.getElementById('normal-board');

    // Clear existing hints
    overlay.innerHTML = '';

    // Get cell size from CSS
    const cellSize = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--cell-size'));
    const gap = 2;
    const padding = 8;

    // Create hint marker
    const marker = document.createElement('div');
    marker.className = 'hint-marker';
    marker.style.left = `${padding + suggestion.y * (cellSize + gap)}px`;
    marker.style.top = `${padding + suggestion.x * (cellSize + gap)}px`;
    marker.innerHTML = suggestion.tool === 0 ? '🖌️' : '💨';
    marker.title = `AI suggests: ${suggestion.tool_name} at (${suggestion.x}, ${suggestion.y})`;

    overlay.appendChild(marker);

    // Add tooltip with explanation
    const tooltip = document.createElement('div');
    tooltip.className = 'hint-tooltip';
    tooltip.innerHTML = `
        <div class="hint-text">
            AI suggests using <strong>${suggestion.tool_name}</strong> here
        </div>
    `;
    tooltip.style.left = `${padding + suggestion.y * (cellSize + gap) + cellSize + 10}px`;
    tooltip.style.top = `${padding + suggestion.x * (cellSize + gap)}px`;
    overlay.appendChild(tooltip);

    // Clear hint after 5 seconds
    if (currentHintTimeout) {
        clearTimeout(currentHintTimeout);
    }
    currentHintTimeout = setTimeout(() => {
        overlay.innerHTML = '';
    }, 5000);
}

// ===========================================
// Add CSS for hint tooltip
// ===========================================

const hintStyles = document.createElement('style');
hintStyles.textContent = `
    .hint-tooltip {
        position: absolute;
        background: rgba(6, 182, 212, 0.9);
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        white-space: nowrap;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 100;
        animation: tooltipFade 0.3s ease;
    }
    
    .hint-tooltip::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 50%;
        transform: translateY(-50%);
        border: 8px solid transparent;
        border-right-color: rgba(6, 182, 212, 0.9);
        border-left: none;
    }
    
    @keyframes tooltipFade {
        from { opacity: 0; transform: translateX(10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .hint-text strong {
        color: #fbbf24;
    }
`;
document.head.appendChild(hintStyles);

// ===========================================
// Keyboard Shortcuts
// ===========================================

document.addEventListener('keydown', (e) => {
    // Don't handle if modal is open
    if (document.querySelector('.modal.active')) return;

    // Tool switching with 1 and 2 keys
    if (e.key === '1') {
        GameState.selectedTool = 0;
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tool-btn[data-tool="0"]').forEach(b => b.classList.add('active'));
    } else if (e.key === '2') {
        GameState.selectedTool = 1;
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tool-btn[data-tool="1"]').forEach(b => b.classList.add('active'));
    }

    // H key for hint in normal mode
    if (e.key === 'h' || e.key === 'H') {
        if (GameState.mode === 'normal') {
            getAIHint();
        }
    }

    // Escape to close modals
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(m => m.classList.remove('active'));
    }
});

// ===========================================
// Event Listeners
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    // Hint button
    document.getElementById('hint-btn').addEventListener('click', getAIHint);
});

// ===========================================
// Compete Mode AI Animation
// ===========================================

// Enhanced AI action visualization for compete mode
function animateAIAction(x, y, tool, callback) {
    const overlay = document.getElementById('ai-action-overlay');
    const cellSize = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--cell-size'));
    const gap = 2;
    const padding = 8;

    // Create action indicator
    const indicator = document.createElement('div');
    indicator.className = 'ai-thinking-indicator';
    indicator.style.left = `${padding + y * (cellSize + gap)}px`;
    indicator.style.top = `${padding + x * (cellSize + gap)}px`;
    indicator.style.width = `${cellSize}px`;
    indicator.style.height = `${cellSize}px`;

    overlay.appendChild(indicator);

    // Animate thinking, then action
    setTimeout(() => {
        indicator.remove();
        showAIAction(x, y, tool);
        if (callback) callback();
    }, 200);
}

// Add styles for AI thinking indicator
const aiStyles = document.createElement('style');
aiStyles.textContent = `
    .ai-thinking-indicator {
        position: absolute;
        border-radius: 6px;
        background: rgba(245, 158, 11, 0.3);
        border: 2px dashed var(--accent-gold);
        animation: aiThink 0.2s ease;
    }
    
    @keyframes aiThink {
        0% { transform: scale(0.8); opacity: 0; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    .ai-action-marker {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
`;
document.head.appendChild(aiStyles);

// ===========================================
// Score Animation
// ===========================================

function animateScoreChange(elementId, oldScore, newScore) {
    const element = document.getElementById(elementId);
    const diff = newScore - oldScore;

    if (diff <= 0) {
        element.textContent = Math.round(newScore);
        return;
    }

    // Create floating score indicator
    const floater = document.createElement('div');
    floater.className = 'score-floater';
    floater.textContent = `+${Math.round(diff)}`;

    const rect = element.getBoundingClientRect();
    floater.style.left = `${rect.left + rect.width / 2}px`;
    floater.style.top = `${rect.top}px`;

    document.body.appendChild(floater);

    // Animate counting up
    const duration = 500;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = 1 - Math.pow(1 - progress, 3);

        const currentScore = oldScore + (diff * easedProgress);
        element.textContent = Math.round(currentScore);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);

    // Remove floater after animation
    setTimeout(() => floater.remove(), 1000);
}

// Add score floater styles
const scoreStyles = document.createElement('style');
scoreStyles.textContent = `
    .score-floater {
        position: fixed;
        font-size: 1.5rem;
        font-weight: 700;
        color: #10b981;
        pointer-events: none;
        transform: translateX(-50%);
        animation: scoreFloat 1s ease forwards;
        z-index: 1000;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    @keyframes scoreFloat {
        0% { opacity: 1; transform: translateX(-50%) translateY(0); }
        100% { opacity: 0; transform: translateX(-50%) translateY(-30px); }
    }
`;
document.head.appendChild(scoreStyles);
