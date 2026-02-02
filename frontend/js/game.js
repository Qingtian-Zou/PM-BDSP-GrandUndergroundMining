/**
 * Grand Underground Mining Game - Main Game Logic
 */

// ===========================================
// Game State
// ===========================================

const GameState = {
    mode: null, // 'normal' or 'compete'
    gameId: null,
    sessionId: null,
    boardShape: [10, 13],
    selectedTool: 0, // 0 = brush, 1 = blower
    isGameOver: false,

    // Normal mode state
    normalState: null,
    normalScore: 0,

    // Compete mode state
    playerState: null,
    playerScore: 0,
    aiState: null,
    aiScore: 0,

    // Items info
    rewards: [],
};

// ===========================================
// API Functions
// ===========================================

const API = {
    // Use CONFIG.API_BASE_URL from config.js
    get baseUrl() {
        return window.CONFIG ? window.CONFIG.API_BASE_URL : '';
    },

    async newNormalGame() {
        const response = await fetch(`${this.baseUrl}/api/game/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        return response.json();
    },

    async getGameState(gameId) {
        const response = await fetch(`${this.baseUrl}/api/game/${gameId}/state`);
        return response.json();
    },

    async takeAction(gameId, x, y, tool) {
        const response = await fetch(`${this.baseUrl}/api/game/${gameId}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y, tool }),
        });
        return response.json();
    },

    async getSuggestion(gameId) {
        const response = await fetch(`${this.baseUrl}/api/game/${gameId}/suggest`);
        return response.json();
    },

    async newCompeteGame() {
        const response = await fetch(`${this.baseUrl}/api/compete/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        return response.json();
    },

    async getCompeteState(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/compete/${sessionId}/state`);
        return response.json();
    },

    async playerAction(sessionId, x, y, tool) {
        const response = await fetch(`${this.baseUrl}/api/compete/${sessionId}/player-action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y, tool }),
        });
        return response.json();
    },

    async aiAction(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/compete/${sessionId}/ai-action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        return response.json();
    },

    async aiFastForward(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/compete/${sessionId}/ai-fast-forward`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        return response.json();
    },
};

// ===========================================
// Board Rendering
// ===========================================

function createBoard(containerId, rows, cols) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${cols}, var(--cell-size))`;
    container.style.gridTemplateRows = `repeat(${rows}, var(--cell-size))`;

    for (let x = 0; x < rows; x++) {
        for (let y = 0; y < cols; y++) {
            const cell = document.createElement('div');
            cell.className = 'cell dust-6';
            cell.dataset.x = x;
            cell.dataset.y = y;
            container.appendChild(cell);
        }
    }
}

function renderBoard(containerId, dust, rewards, retrieved) {
    const container = document.getElementById(containerId);
    const cells = container.querySelectorAll('.cell');
    const [rows, cols] = GameState.boardShape;

    // Create item mask lookup
    const itemMask = {};
    rewards.forEach((reward, idx) => {
        for (let x = 0; x < rows; x++) {
            for (let y = 0; y < cols; y++) {
                if (reward.mask[x][y]) {
                    itemMask[`${x},${y}`] = {
                        itemIdx: idx,
                        name: reward.name,
                        retrieved: reward.retrieved,
                    };
                }
            }
        }
    });

    cells.forEach(cell => {
        const x = parseInt(cell.dataset.x);
        const y = parseInt(cell.dataset.y);
        const dustLevel = dust[x][y];

        // Remove old classes
        cell.className = 'cell';
        cell.innerHTML = '';

        // Add dust class
        cell.classList.add(`dust-${dustLevel}`);

        // Check for items
        const key = `${x},${y}`;
        if (itemMask[key] && dustLevel === 0) {
            cell.classList.add('has-item');
            const indicator = document.createElement('div');
            indicator.className = `item-indicator ${getItemClass(itemMask[key].name)}`;
            cell.appendChild(indicator);
        }
    });
}

function getItemClass(itemName) {
    const name = itemName.toLowerCase();
    if (name.includes('fossil') || name.includes('amber')) return 'item-fossil';
    if (name.includes('sphere')) return 'item-sphere';
    if (name.includes('red')) return 'item-shard-red';
    if (name.includes('blue')) return 'item-shard-blue';
    if (name.includes('green')) return 'item-shard-green';
    if (name.includes('yellow')) return 'item-shard-yellow';
    if (name.includes('shiny')) return 'item-statue-shiny';
    if (name.includes('statue')) return 'item-statue';
    return 'item-special';
}

function getItemEmoji(itemName) {
    const name = itemName.toLowerCase();
    if (name.includes('fossil')) return '🦴';
    if (name.includes('amber')) return '🟠';
    if (name.includes('sphere')) return '⚪';
    if (name.includes('shard')) return '💎';
    if (name.includes('statue')) return '🗿';
    if (name.includes('star')) return '⭐';
    if (name.includes('bone')) return '🦴';
    if (name.includes('revive')) return '💊';
    return '✨';
}

// ===========================================
// UI Updates
// ===========================================

function updateEnergy(prefix, energy, maxEnergy = 95) {
    const fill = document.getElementById(`${prefix}-energy-fill`);
    const text = document.getElementById(`${prefix}-energy-text`);

    const percent = (energy / maxEnergy) * 100;
    fill.style.width = `${percent}%`;
    text.textContent = energy;

    // Update color based on energy level
    fill.classList.remove('low', 'medium');
    if (percent < 25) {
        fill.classList.add('low');
    } else if (percent < 50) {
        fill.classList.add('medium');
    }
}

function updateScore(elementId, score) {
    document.getElementById(elementId).textContent = Math.round(score);
}

function renderItemsList(containerId, rewards) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    rewards.forEach((reward, idx) => {
        const card = document.createElement('div');
        card.className = `item-card ${reward.retrieved ? 'retrieved' : ''}`;
        card.innerHTML = `
            <div class="item-icon ${getItemClass(reward.name)}">
                ${getItemEmoji(reward.name)}
            </div>
            <div class="item-info">
                <div class="item-name">${reward.name}</div>
                <div class="item-value">+${reward.value} pts</div>
            </div>
        `;
        container.appendChild(card);
    });
}

// ===========================================
// Screen Management
// ===========================================

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function showModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function hideModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// ===========================================
// Game Logic
// ===========================================

async function startNormalGame() {
    GameState.mode = 'normal';
    GameState.isGameOver = false;

    const data = await API.newNormalGame();
    GameState.gameId = data.game_id;
    GameState.boardShape = data.board_shape;
    GameState.normalState = data.state;
    GameState.normalScore = 0;
    GameState.rewards = data.state.rewards;

    // Create and render board
    createBoard('normal-board', ...GameState.boardShape);
    renderBoard('normal-board', data.state.dust, data.state.rewards, data.state.retrieved);

    // Update UI
    updateEnergy('normal', data.state.energy);
    updateScore('normal-score', 0);
    renderItemsList('normal-items-list', data.state.rewards);

    // Setup click handlers
    setupBoardClickHandler('normal-board', handleNormalClick);

    showScreen('normal-game');
}

async function startCompeteGame() {
    GameState.mode = 'compete';
    GameState.isGameOver = false;

    const data = await API.newCompeteGame();
    GameState.sessionId = data.session_id;
    GameState.boardShape = data.board_shape;
    GameState.playerState = data.player_state;
    GameState.aiState = data.ai_state;
    GameState.playerScore = 0;
    GameState.aiScore = 0;
    GameState.rewards = data.player_state.rewards;

    // Create and render boards
    createBoard('player-board', ...GameState.boardShape);
    createBoard('ai-board', ...GameState.boardShape);

    renderBoard('player-board', data.player_state.dust, data.player_state.rewards, data.player_state.retrieved);
    renderBoard('ai-board', data.ai_state.dust, data.ai_state.rewards, data.ai_state.retrieved);

    // Update UI
    updateEnergy('player', data.player_state.energy);
    updateEnergy('ai', data.ai_state.energy);
    updateScore('player-score', 0);
    updateScore('ai-score', 0);
    renderItemsList('compete-items-list', data.player_state.rewards);

    // Reset AI status
    updateAIStatus('waiting');

    // Setup click handlers
    setupBoardClickHandler('player-board', handleCompeteClick);

    showScreen('compete-game');
}

function setupBoardClickHandler(boardId, handler) {
    const board = document.getElementById(boardId);
    board.onclick = (e) => {
        const cell = e.target.closest('.cell');
        if (cell) {
            const x = parseInt(cell.dataset.x);
            const y = parseInt(cell.dataset.y);
            handler(x, y);
        }
    };
}

async function handleNormalClick(x, y) {
    if (GameState.isGameOver) return;

    const data = await API.takeAction(GameState.gameId, x, y, GameState.selectedTool);

    GameState.normalState = data.state;
    GameState.normalScore = data.score;
    GameState.rewards = data.state.rewards;

    renderBoard('normal-board', data.state.dust, data.state.rewards, data.state.retrieved);
    updateEnergy('normal', data.state.energy);
    updateScore('normal-score', data.score);
    renderItemsList('normal-items-list', data.state.rewards);

    // Clear any hint
    document.getElementById('normal-hint-overlay').innerHTML = '';

    if (data.done) {
        endNormalGame();
    }
}

async function handleCompeteClick(x, y) {
    if (GameState.isGameOver) return;
    if (GameState.playerState && GameState.playerState.done) return;

    // Player takes action
    const playerData = await API.playerAction(GameState.sessionId, x, y, GameState.selectedTool);

    GameState.playerState = playerData.player.state;
    GameState.playerScore = playerData.player.score;

    renderBoard('player-board', playerData.player.state.dust, playerData.player.state.rewards, playerData.player.state.retrieved);
    updateEnergy('player', playerData.player.state.energy);
    updateScore('player-score', playerData.player.score);
    renderItemsList('compete-items-list', playerData.player.state.rewards);

    // Check if player finished
    if (playerData.player.done) {
        // Let AI finish
        updateAIStatus('finishing');
        await autoFinishAI();
        endCompeteGame();
        return;
    }

    // AI takes action
    updateAIStatus('thinking');
    await new Promise(r => setTimeout(r, 300)); // Small delay for effect

    const aiData = await API.aiAction(GameState.sessionId);

    if (aiData.ai.action) {
        // Show AI's action briefly
        showAIAction(aiData.ai.action.x, aiData.ai.action.y, aiData.ai.action.tool);
    }

    GameState.aiState = aiData.ai.state || GameState.aiState;
    GameState.aiScore = aiData.ai.score;

    if (aiData.ai.state) {
        renderBoard('ai-board', aiData.ai.state.dust, aiData.ai.state.rewards, aiData.ai.state.retrieved);
    }
    updateEnergy('ai', aiData.ai.state ? aiData.ai.state.energy : 0);
    updateScore('ai-score', aiData.ai.score);

    if (aiData.ai.done) {
        updateAIStatus('done');
    } else {
        updateAIStatus('waiting');
    }

    // Check if both finished
    if (playerData.player.done && aiData.ai.done) {
        endCompeteGame();
    }
}

function showAIAction(x, y, tool) {
    const overlay = document.getElementById('ai-action-overlay');
    const board = document.getElementById('ai-board');
    const cellSize = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--cell-size'));

    const marker = document.createElement('div');
    marker.className = 'ai-action-marker';
    marker.style.left = `${8 + y * (cellSize + 2)}px`;
    marker.style.top = `${8 + x * (cellSize + 2)}px`;
    marker.innerHTML = tool === 0 ? '🖌️' : '💨';

    overlay.appendChild(marker);

    setTimeout(() => marker.remove(), 600);
}

function updateAIStatus(status) {
    const container = document.getElementById('ai-status');
    const icon = container.querySelector('.status-icon');
    const text = container.querySelector('.status-text');

    container.classList.remove('done');

    switch (status) {
        case 'waiting':
            icon.textContent = '⏳';
            text.textContent = 'Waiting for your move...';
            break;
        case 'thinking':
            icon.textContent = '🤔';
            text.textContent = 'AI is thinking...';
            break;
        case 'finishing':
            icon.textContent = '⚡';
            text.textContent = 'AI is finishing up...';
            break;
        case 'done':
            icon.textContent = '✅';
            text.textContent = 'AI has finished!';
            container.classList.add('done');
            break;
    }
}

async function autoFinishAI() {
    // Let AI take actions until it's done
    while (true) {
        const aiData = await API.aiAction(GameState.sessionId);

        if (aiData.ai.action) {
            showAIAction(aiData.ai.action.x, aiData.ai.action.y, aiData.ai.action.tool);
            await new Promise(r => setTimeout(r, 100)); // Fast animation
        }

        GameState.aiState = aiData.ai.state || GameState.aiState;
        GameState.aiScore = aiData.ai.score;

        if (aiData.ai.state) {
            renderBoard('ai-board', aiData.ai.state.dust, aiData.ai.state.rewards, aiData.ai.state.retrieved);
        }
        updateEnergy('ai', aiData.ai.state ? aiData.ai.state.energy : 0);
        updateScore('ai-score', aiData.ai.score);

        if (aiData.ai.done) {
            updateAIStatus('done');
            break;
        }
    }
}

async function fastForwardAI() {
    hideModal('fast-forward-modal');
    updateAIStatus('finishing');

    const data = await API.aiFastForward(GameState.sessionId);

    GameState.aiState = data.ai.final_state;
    GameState.aiScore = data.ai.score;

    renderBoard('ai-board', data.ai.final_state.dust, data.ai.final_state.rewards, data.ai.final_state.retrieved);
    updateEnergy('ai', data.ai.final_state.energy);
    updateScore('ai-score', data.ai.score);
    updateAIStatus('done');

    // End game if player is also done
    if (GameState.playerState && GameState.playerState.done) {
        endCompeteGame();
    }
}

function endNormalGame() {
    GameState.isGameOver = true;

    const retrieved = GameState.rewards.filter(r => r.retrieved).length;
    const total = GameState.rewards.length;

    document.getElementById('final-score').textContent = Math.round(GameState.normalScore);
    document.getElementById('items-retrieved').textContent = retrieved;
    document.getElementById('items-total').textContent = total;

    document.getElementById('single-result').classList.remove('hidden');
    document.getElementById('compete-result').classList.add('hidden');
    document.getElementById('modal-title').textContent = 'Game Over!';

    showModal('game-over-modal');
}

function endCompeteGame() {
    GameState.isGameOver = true;

    document.getElementById('compete-player-score').textContent = Math.round(GameState.playerScore);
    document.getElementById('compete-ai-score').textContent = Math.round(GameState.aiScore);

    const winnerText = document.getElementById('winner-text');
    if (GameState.playerScore > GameState.aiScore) {
        winnerText.textContent = '🎉 You Win!';
        winnerText.className = 'winner-announcement win';
    } else if (GameState.playerScore < GameState.aiScore) {
        winnerText.textContent = '🤖 AI Wins!';
        winnerText.className = 'winner-announcement lose';
    } else {
        winnerText.textContent = "🤝 It's a Tie!";
        winnerText.className = 'winner-announcement tie';
    }

    document.getElementById('single-result').classList.add('hidden');
    document.getElementById('compete-result').classList.remove('hidden');
    document.getElementById('modal-title').textContent = 'Match Complete!';

    showModal('game-over-modal');
}

// ===========================================
// Event Listeners
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    // Mode selection
    document.querySelectorAll('.mode-card').forEach(card => {
        card.addEventListener('click', () => {
            const mode = card.dataset.mode;
            if (mode === 'normal') {
                startNormalGame();
            } else if (mode === 'compete') {
                startCompeteGame();
            }
        });
    });

    // New game button
    document.getElementById('new-game-btn').addEventListener('click', () => {
        showScreen('mode-selection');
    });

    // Tool selection - Normal mode
    document.getElementById('brush-btn').addEventListener('click', (e) => {
        GameState.selectedTool = 0;
        document.querySelectorAll('#normal-game .tool-btn').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
    });

    document.getElementById('blower-btn').addEventListener('click', (e) => {
        GameState.selectedTool = 1;
        document.querySelectorAll('#normal-game .tool-btn').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
    });

    // Tool selection - Compete mode
    document.getElementById('compete-brush-btn').addEventListener('click', (e) => {
        GameState.selectedTool = 0;
        document.querySelectorAll('#compete-game .tool-btn').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
    });

    document.getElementById('compete-blower-btn').addEventListener('click', (e) => {
        GameState.selectedTool = 1;
        document.querySelectorAll('#compete-game .tool-btn').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
    });

    // Fast forward button
    document.getElementById('fast-forward-btn').addEventListener('click', () => {
        showModal('fast-forward-modal');
    });

    document.getElementById('cancel-ff-btn').addEventListener('click', () => {
        hideModal('fast-forward-modal');
    });

    document.getElementById('confirm-ff-btn').addEventListener('click', () => {
        fastForwardAI();
    });

    // Game over modal
    document.getElementById('play-again-btn').addEventListener('click', () => {
        hideModal('game-over-modal');
        if (GameState.mode === 'normal') {
            startNormalGame();
        } else {
            startCompeteGame();
        }
    });

    document.getElementById('change-mode-btn').addEventListener('click', () => {
        hideModal('game-over-modal');
        showScreen('mode-selection');
    });
});
