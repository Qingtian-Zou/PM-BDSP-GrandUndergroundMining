"""
Flask Server for Grand Underground Mining Game - Backend API

Provides REST API for game management and AI assistance.
Designed for deployment on Railway.
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import torch
import numpy as np
from pathlib import Path

from game import MiningEnv
from agents.smart_agent import SmartAgent

app = Flask(__name__)

# Configure CORS to allow requests from GitHub Pages
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
CORS(app, origins=ALLOWED_ORIGINS)

# Game session storage
games = {}
compete_sessions = {}

# Configuration
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "smart_20260201_175126" / "best_model.pt"
BOARD_SHAPE = (10, 13)
MAX_ENERGY = 95

# Global agent (loaded once)
ai_agent = None


def remap_state_dict(state_dict, expected_keys):
    """Remap state dict keys to match expected model architecture."""
    # Build mapping based on key patterns
    remapped = {}
    
    state_keys = sorted([k for k in state_dict.keys() if 'weight' in k or 'bias' in k])
    expected_sorted = sorted(expected_keys)
    
    # Group by prefix (conv, fc, action_conv, etc.)
    def group_keys(keys):
        groups = {}
        for k in keys:
            parts = k.split('.')
            prefix = parts[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(k)
        return groups
    
    state_groups = group_keys(state_keys)
    expected_groups = group_keys(expected_sorted)
    
    for prefix in expected_groups:
        if prefix not in state_groups:
            continue
        
        state_prefix_keys = sorted(state_groups[prefix])
        expected_prefix_keys = sorted(expected_groups[prefix])
        
        # Match by position (weights/biases alternate)
        if len(state_prefix_keys) == len(expected_prefix_keys):
            for sk, ek in zip(state_prefix_keys, expected_prefix_keys):
                if state_dict[sk].shape == state_dict[sk].shape:  # Basic shape compatibility
                    remapped[ek] = state_dict[sk]
    
    return remapped


def get_agent():
    """Lazy load the AI agent."""
    global ai_agent
    if ai_agent is None:
        if CHECKPOINT_PATH.exists():
            # Load checkpoint first to read the config
            checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
            
            # Get config from checkpoint, with fallback defaults
            config = checkpoint.get('config', {})
            conv_channels = config.get('conv_channels', 128)
            n_conv_layers = config.get('n_conv_layers', 4)
            fc_hidden_size = config.get('fc_hidden_size', 64)
            
            print(f"Loading model with config: conv_channels={conv_channels}, n_conv_layers={n_conv_layers}, fc_hidden_size={fc_hidden_size}")
            
            ai_agent = SmartAgent(
                board_shape=BOARD_SHAPE,
                n_retrieved=4,
                conv_channels=conv_channels,
                n_conv_layers=n_conv_layers,
                fc_hidden_size=fc_hidden_size,
            )
            
            # Load weights
            ai_agent.policy.load_state_dict(checkpoint["policy"])
            ai_agent.value.load_state_dict(checkpoint["value"])
            print(f"Loaded AI model from {CHECKPOINT_PATH}")
        else:
            print(f"Warning: No checkpoint found at {CHECKPOINT_PATH}, using untrained agent with default config")
            ai_agent = SmartAgent(
                board_shape=BOARD_SHAPE,
                n_retrieved=4,
                conv_channels=128,
                n_conv_layers=4,
                fc_hidden_size=64,
            )
    return ai_agent


def state_to_dict(state, reward_info=None):
    """Convert environment state to JSON-serializable dict."""
    result = {
        "dust": state["dust"].tolist(),
        "energy": int(state["energy"]),
        "retrieved": state["retrieved"].tolist(),
    }
    if reward_info:
        result["rewards"] = reward_info
    return result


def get_reward_info(env):
    """Extract reward item information from environment."""
    rewards = []
    for i, ((name, (x, y)), mask, value) in enumerate(zip(env.rewards, env.reward_masks, env.reward_values)):
        # Get shape dimensions
        shape_coords = np.argwhere(mask)
        rewards.append({
            "id": i,
            "name": name,
            "value": value,
            "position": [int(x), int(y)],
            "mask": mask.tolist(),
            "retrieved": bool(env.retrieved[i]),
        })
    return rewards


class GameSession:
    """Manages a single game session."""
    
    def __init__(self, seed=None):
        self.env = MiningEnv(
            board_shape=BOARD_SHAPE,
            max_energy=MAX_ENERGY,
            minor_rewards=0.7,
        )
        self.state, _ = self.env.reset(seed=seed)
        self.score = 0
        self.done = False
        self.actions_taken = []
        self.seed = seed
    
    def take_action(self, x, y, tool):
        """Execute an action and return the result."""
        if self.done:
            return None, 0, True
        
        action = (x, y, tool)
        next_state, reward, done, _, _ = self.env.step(action)
        self.state = next_state
        self.score += reward
        self.done = bool(done)  # Convert numpy bool to Python bool
        self.actions_taken.append(action)
        
        return next_state, float(reward), bool(done)
    
    def get_ai_suggestion(self):
        """Get AI's recommended action."""
        agent = get_agent()
        action, _, _ = agent.select_action(self.state, training=False)
        return action


class CompeteSession:
    """Manages a compete mode session with two games."""
    
    def __init__(self):
        # Use same seed for both games
        self.seed = np.random.randint(0, 2**31)
        self.player_game = GameSession(seed=self.seed)
        self.ai_game = GameSession(seed=self.seed)
        self.player_turn = True  # True if waiting for player action
        self.ai_finished = False
        self.player_finished = False
    
    def player_action(self, x, y, tool):
        """Player takes an action."""
        return self.player_game.take_action(x, y, tool)
    
    def ai_action(self):
        """AI takes one action based on its own game state."""
        if self.ai_game.done:
            return None, 0, True
        
        action = self.ai_game.get_ai_suggestion()
        return self.ai_game.take_action(*action)
    
    def ai_fast_forward(self):
        """AI completes its entire game."""
        actions = []
        while not self.ai_game.done:
            action = self.ai_game.get_ai_suggestion()
            _, reward, done = self.ai_game.take_action(*action)
            actions.append({
                "action": list(action),
                "reward": reward,
            })
        return actions


# ===================
# Health Check
# ===================

@app.route('/')
def health_check():
    """Health check endpoint for Railway."""
    return jsonify({
        "status": "healthy",
        "service": "Grand Underground Mining API",
        "version": "1.0.0"
    })


@app.route('/api/health')
def api_health():
    """API health check."""
    return jsonify({"status": "ok"})


# ===================
# Game API Routes
# ===================

@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Create a new normal mode game."""
    game_id = str(uuid.uuid4())
    games[game_id] = GameSession()
    
    game = games[game_id]
    return jsonify({
        "game_id": game_id,
        "state": state_to_dict(game.state, get_reward_info(game.env)),
        "board_shape": list(BOARD_SHAPE),
    })


@app.route('/api/game/<game_id>/state', methods=['GET'])
def get_state(game_id):
    """Get current game state."""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404
    
    game = games[game_id]
    return jsonify({
        "state": state_to_dict(game.state, get_reward_info(game.env)),
        "score": game.score,
        "done": game.done,
    })


@app.route('/api/game/<game_id>/action', methods=['POST'])
def take_action(game_id):
    """Execute player action."""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404
    
    data = request.json
    x, y, tool = data['x'], data['y'], data['tool']
    
    game = games[game_id]
    next_state, reward, done = game.take_action(x, y, tool)
    
    return jsonify({
        "state": state_to_dict(next_state, get_reward_info(game.env)),
        "reward": reward,
        "score": game.score,
        "done": done,
    })


@app.route('/api/game/<game_id>/suggest', methods=['GET'])
def get_suggestion(game_id):
    """Get AI suggestion for current state."""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404
    
    game = games[game_id]
    if game.done:
        return jsonify({"error": "Game is already over"}), 400
    
    action = game.get_ai_suggestion()
    
    return jsonify({
        "suggestion": {
            "x": action[0],
            "y": action[1],
            "tool": action[2],
            "tool_name": "brush" if action[2] == 0 else "blower",
        }
    })


# ===================
# Compete Mode Routes
# ===================

@app.route('/api/compete/new', methods=['POST'])
def new_compete():
    """Create a new compete mode session."""
    session_id = str(uuid.uuid4())
    compete_sessions[session_id] = CompeteSession()
    
    session = compete_sessions[session_id]
    return jsonify({
        "session_id": session_id,
        "player_state": state_to_dict(
            session.player_game.state, 
            get_reward_info(session.player_game.env)
        ),
        "ai_state": state_to_dict(
            session.ai_game.state,
            get_reward_info(session.ai_game.env)
        ),
        "board_shape": list(BOARD_SHAPE),
    })


@app.route('/api/compete/<session_id>/state', methods=['GET'])
def get_compete_state(session_id):
    """Get current compete session state."""
    if session_id not in compete_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = compete_sessions[session_id]
    return jsonify({
        "player": {
            "state": state_to_dict(
                session.player_game.state,
                get_reward_info(session.player_game.env)
            ),
            "score": session.player_game.score,
            "done": session.player_game.done,
        },
        "ai": {
            "state": state_to_dict(
                session.ai_game.state,
                get_reward_info(session.ai_game.env)
            ),
            "score": session.ai_game.score,
            "done": session.ai_game.done,
        },
    })


@app.route('/api/compete/<session_id>/player-action', methods=['POST'])
def compete_player_action(session_id):
    """Player takes action in compete mode."""
    if session_id not in compete_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    data = request.json
    x, y, tool = data['x'], data['y'], data['tool']
    
    session = compete_sessions[session_id]
    next_state, reward, done = session.player_action(x, y, tool)
    
    return jsonify({
        "player": {
            "state": state_to_dict(next_state, get_reward_info(session.player_game.env)),
            "reward": reward,
            "score": session.player_game.score,
            "done": done,
        }
    })


@app.route('/api/compete/<session_id>/ai-action', methods=['POST'])
def compete_ai_action(session_id):
    """AI takes one action in compete mode."""
    if session_id not in compete_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = compete_sessions[session_id]
    next_state, reward, done = session.ai_action()
    
    if next_state is None:
        return jsonify({
            "ai": {
                "done": True,
                "score": session.ai_game.score,
            }
        })
    
    action = session.ai_game.actions_taken[-1]
    return jsonify({
        "ai": {
            "action": {"x": action[0], "y": action[1], "tool": action[2]},
            "state": state_to_dict(next_state, get_reward_info(session.ai_game.env)),
            "reward": reward,
            "score": session.ai_game.score,
            "done": done,
        }
    })


@app.route('/api/compete/<session_id>/ai-fast-forward', methods=['POST'])
def compete_ai_fast_forward(session_id):
    """AI completes its entire game instantly."""
    if session_id not in compete_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = compete_sessions[session_id]
    actions = session.ai_fast_forward()
    
    return jsonify({
        "ai": {
            "actions": actions,
            "final_state": state_to_dict(
                session.ai_game.state,
                get_reward_info(session.ai_game.env)
            ),
            "score": session.ai_game.score,
            "done": True,
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"Starting Grand Underground Mining Game API Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
