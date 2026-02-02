# PM-BDSP-GrandUndergroundMining

A reinforcement learning framework and **playable web game** for the mining minigame from Pokémon Brilliant Diamond & Shining Pearl's Grand Underground. Play in your browser against a trained AI agent, or use the customizable [Gymnasium](https://gymnasium.farama.org/)-compatible environment with multiple deep RL implementations for research and experimentation.

## 🎮 Live Demo

**[Play the Game Online →](https://qingtian-zou.github.io/PM-BDSP-GrandUndergroundMining/)**

> ⚠️ **Note:** The live demo runs on free-tier hosting (Railway). The backend may be slow to respond initially (cold start) or become temporarily unavailable due to resource limitations. For the best experience, we recommend running the game locally.

---

## ✨ Features

### Custom RL Environment (`game/mining_env.py`)
- **Rectangular grid board** with dust layers (2-6) and randomly placed, non-overlapping rewards
- **18 unique reward types** defined in [`game/minable.json`](game/minable.json) including fossils, shards, spheres, and statues
- **Two mining tools:**
  - **Brush:** Clears 2 dust on center, 1 on 4-neighbors. Costs 2 energy.
  - **Blower:** Clears 2 on center and 4-neighbors, 1 on diagonals. Costs 4 energy.
- **Flexible reward system:**
  - Major rewards for fully uncovering items
  - Configurable minor rewards for partial uncovering (reward shaping)
- Episode ends when energy depletes or all rewards are retrieved

### Multiple RL Agent Implementations

| Agent | File | Algorithm | Key Features |
|-------|------|-----------|--------------|
| **DQN Agent** | `agents/dqn_agent.py` | Dueling Double DQN | Prioritized Experience Replay, Sum Tree sampling |
| **PPO Agent** | `agents/ppo_agent.py` | Proximal Policy Optimization | Actor-Critic architecture, GAE, clipped objective |
| **Optimized DQN** | `agents/optimized_agent.py` | Enhanced DQN | Soft target updates (τ=0.005), N-step returns, curiosity bonus |
| **Smart Agent** | `agents/smart_agent.py` | Policy Gradient | Distance-based reward shaping, focused action masking |

### Training Scripts

| Script | Agent | Description |
|--------|-------|-------------|
| `training/train_agent.py` | DQN | Standard DQN training with evaluation |
| `training/train_ppo.py` | PPO | PPO training with timestep-based updates |
| `training/train_optimized.py` | Optimized DQN | Enhanced DQN with better hyperparameters |
| `training/train_smart.py` | Smart Agent | Training with reward shaping and action masking |

### Hyperparameter Search (`training/hyperparam_search.py`)
- **Parallel grid search** across hyperparameter combinations
- **Configurable search space:** learning rate, gamma, entropy coefficient, network architecture
- **Automatic logging** to CSV and JSON
- **Analysis tools** for parameter importance and top configurations

### 🎮 Playable Web Game (`server.py`)

A polished browser-based game built with Flask that lets you play or compete against the trained AI:

**Normal Mode:**
- Play manually with full control over mining strategy
- **Get AI Hint** button provides move suggestions from the trained agent
- Track your score and collected items in real-time

**Compete Mode:**
- Race against the AI on identical boards (same seed)
- Turn-based gameplay: you move, then AI responds
- **Fast Forward** option lets AI finish instantly (with forfeit warning)
- Compare final scores to see who mines better!

**Visual Features:**
- Dark underground cave theme with polished UI
- Distinct dust layer textures:
  - **Rock (levels 5-6):** Dark brown with texture
  - **Hard Soil (levels 3-4):** Medium brown
  - **Soft Soil (levels 1-2):** Light tan/beige
- Crack patterns on odd levels (1, 3, 5) for visual distinction
- Smooth animations and glassmorphism effects

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `torch`
- `matplotlib`
- `flask` (for web game)

### Installation

```bash
pip install gymnasium numpy torch matplotlib flask flask-cors
```

### Play the Game

Start the web server and play in your browser:

```bash
python server.py
```

Then open http://localhost:5000 to:
- **Normal Mode:** Play solo with optional AI hints
- **Compete Mode:** Challenge the trained AI agent

### Environment Usage

```python
from game import MiningEnv

env = MiningEnv(
    board_shape=(12, 16),
    max_energy=120,
    minor_rewards=0.5  # Reward shaping coefficient
)
obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print("Reward:", reward)
```

### Training Agents

**Smart Agent (Recommended):**
```bash
# Best configuration from hyperparameter search
python training/train_smart.py --episodes 2000 --lr 0.001 --gamma 0.9 \
    --entropy-coef 0.01 --minor-rewards 0.7 \
    --conv-channels 128 --n-conv-layers 4 --fc-hidden-size 64
```

**DQN Agent:**
```bash
python training/train_agent.py --episodes 5000 --eval-freq 100
```

**PPO Agent:**
```bash
python training/train_ppo.py --timesteps 500000 --eval-freq 10000
```

**Optimized DQN:**
```bash
python training/train_optimized.py --episodes 3000 --eval-freq 100
```

### Hyperparameter Search

Run parallel grid search to find optimal hyperparameters:

```bash
python training/hyperparam_search.py --workers 4 --episodes 1000 --eval-episodes 100
```

Results are saved to `hyperparam_search/search_<timestamp>/` including:
- `results.csv` - All configurations with metrics
- `grid_search_results.json` - Detailed results
- `best_config.sh` - Script to reproduce best configuration

---

## 📊 Experimental Results

### Best Configuration Found

From hyperparameter search with **2187 configurations** tested:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Gamma | 0.9 |
| Entropy Coefficient | 0.01 |
| Minor Rewards | 0.7 |
| Conv Channels | 128 |
| Conv Layers | 4 |
| FC Hidden Size | 64 |
| **Eval Reward** | **19.05 ± 10.7** |
| **Items Retrieved** | **0.88** |
| Model Size | ~994K parameters |

### Top 5 Configurations

| Rank | Eval Reward | Retrieved | Key Settings |
|------|-------------|-----------|---------------|
| 1 | 19.05 | 0.88 | lr=0.001, γ=0.9, ent=0.01, mr=0.7, 128ch, 4L |
| 2 | 18.00 | 0.80 | lr=0.001, γ=0.9, ent=0.1, mr=0.7, 128ch, 4L |
| 3 | 16.84 | 0.68 | lr=0.001, γ=0.9, ent=0.1, mr=0.7, 128ch, 4L |
| 4 | 16.69 | 0.66 | lr=0.002, γ=0.9, ent=0.01, mr=0.7, 64ch, 4L |
| 5 | 16.59 | 0.64 | lr=0.002, γ=0.9, ent=0.1, mr=0.7, 64ch, 4L |

### Key Findings

- **Reward shaping is crucial:** `minor_rewards=0.7` consistently outperforms lower values
- **Lower gamma works better:** γ=0.9 outperforms γ=0.95 and γ=0.99 for this sparse-reward task
- **Deeper networks help:** 4 conv layers significantly outperform 2-3 layers
- **Larger conv channels improve performance:** 128 channels > 64 > 32
- **Entropy coefficient:** Lower values (0.01) tend to perform best
- **Learning rate:** 0.001 is the sweet spot; 0.002 works but slightly worse

---

## 🎮 Environment Details

### Action Space
`MultiDiscrete([board_height, board_width, 2])` - Tuple `(x, y, tool)` where:
- `x, y`: Grid coordinates
- `tool`: 0 (brush) or 1 (blower)

### Observation Space
- `dust`: 2D array of dust layer values (0-6)
- `energy`: Remaining energy (integer)
- `retrieved`: Binary vector indicating collected rewards

### Reward Items (from `minable.json`)

| Category | Items | Value Range |
|----------|-------|-------------|
| Spheres | S, L | 4-7 |
| Statues | Normal, Shiny | 10-15 |
| Shards | Red, Blue, Green, Yellow | 11-16 |
| Fossils | Helix, Dome, Root, Claw, Armor, Skull | 14-20 |
| Other | Old Amber, Rare Bone, Star Piece, Revive | 9-15 |

---

## 📁 Project Structure

```
├── frontend/              # Static frontend (GitHub Pages)
│   ├── index.html         # Main game page
│   ├── css/style.css      # Game styles
│   ├── js/
│   │   ├── config.js      # API URL configuration
│   │   ├── game.js        # Main game logic
│   │   └── ai-assist.js   # AI hint functionality
│   └── README.md          # Frontend deployment guide
│
├── backend/               # Flask API (Railway)
│   ├── server.py          # REST API server
│   ├── game/              # Game environment module
│   ├── agents/            # RL agent implementations
│   ├── checkpoints/       # Trained model weights
│   ├── requirements.txt   # Python dependencies
│   ├── Procfile           # Railway/Heroku deployment
│   ├── railway.toml       # Railway configuration
│   └── README.md          # Backend deployment guide
│
├── game/                  # Game environment module (development)
│   ├── __init__.py
│   ├── mining_env.py      # Gymnasium environment
│   └── minable.json       # Reward definitions
│
├── agents/                # RL agent implementations (development)
│   ├── __init__.py
│   ├── dqn_agent.py       # Dueling Double DQN + PER
│   ├── ppo_agent.py       # PPO with GAE
│   ├── optimized_agent.py # Enhanced DQN
│   └── smart_agent.py     # Smart agent with reward shaping
│
├── training/              # Training scripts and utilities
│   ├── __init__.py
│   ├── train_agent.py     # DQN training script
│   ├── train_ppo.py       # PPO training script
│   ├── train_optimized.py # Optimized DQN training
│   ├── train_smart.py     # Smart agent training
│   └── hyperparam_search.py # Grid search framework
│
├── static/                # Legacy web UI (for local development)
├── server.py              # Legacy Flask server (local)
├── game.ipynb             # Interactive notebook demo
├── hyperparam_search/     # Search results
└── checkpoints/           # Saved models
```

## 🌐 Deployment

### GitHub Pages (Frontend)
The static frontend is deployed to GitHub Pages for hosting the game UI.

### Railway (Backend)
The Flask API is deployed to Railway for running the AI agent and game logic.

See `frontend/README.md` and `backend/README.md` for detailed deployment instructions.

---

## 💡 Tips for Better Learning

1. **Use reward shaping** (`minor_rewards > 0`) to provide learning signal for partial progress
2. **Normalize inputs** - dust layers and energy are already bounded
3. **Start with Smart Agent** - it's designed for this sparse-reward environment
4. **Run hyperparameter search** to find optimal settings for your hardware
5. **Use GPU** if available - agents auto-detect CUDA

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- Inspired by the mining minigame in Pokémon Brilliant Diamond & Shining Pearl
- Built with [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/)
