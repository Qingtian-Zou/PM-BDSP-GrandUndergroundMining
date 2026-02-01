# PM-BDSP-GrandUndergroundMining

A reinforcement learning framework for simulating and solving the mining minigame from Pokémon Brilliant Diamond & Shining Pearl's Grand Underground. This project provides a customizable [Gymnasium](https://gymnasium.farama.org/)-compatible environment, multiple deep RL agent implementations, and comprehensive hyperparameter search capabilities for research and experimentation.

---

## ✨ Features

### Custom RL Environment (`mining_env.py`)
- **Rectangular grid board** with dust layers (2-6) and randomly placed, non-overlapping rewards
- **18 unique reward types** defined in [`minable.json`](minable.json) including fossils, shards, spheres, and statues
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
| **DQN Agent** | `dqn_agent.py` | Dueling Double DQN | Prioritized Experience Replay, Sum Tree sampling |
| **PPO Agent** | `ppo_agent.py` | Proximal Policy Optimization | Actor-Critic architecture, GAE, clipped objective |
| **Optimized DQN** | `optimized_agent.py` | Enhanced DQN | Soft target updates (τ=0.005), N-step returns, curiosity bonus |
| **Smart Agent** | `smart_agent.py` | Policy Gradient | Distance-based reward shaping, focused action masking |

### Training Scripts

| Script | Agent | Description |
|--------|-------|-------------|
| `train_agent.py` | DQN | Standard DQN training with evaluation |
| `train_ppo.py` | PPO | PPO training with timestep-based updates |
| `train_optimized.py` | Optimized DQN | Enhanced DQN with better hyperparameters |
| `train_smart.py` | Smart Agent | Training with reward shaping and action masking |

### Hyperparameter Search (`hyperparam_search.py`)
- **Parallel grid search** across hyperparameter combinations
- **Configurable search space:** learning rate, gamma, entropy coefficient, network architecture
- **Automatic logging** to CSV and JSON
- **Analysis tools** for parameter importance and top configurations

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `torch`
- `matplotlib`

### Installation

```bash
pip install gymnasium numpy torch matplotlib
```

### Environment Usage

```python
import mining_env

env = mining_env.MiningEnv(
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
python train_smart.py --episodes 2000 --lr 0.001 --gamma 0.9 \
    --entropy-coef 0.01 --minor-rewards 0.7 \
    --conv-channels 128 --n-conv-layers 4 --fc-hidden-size 64
```

**DQN Agent:**
```bash
python train_agent.py --episodes 5000 --eval-freq 100
```

**PPO Agent:**
```bash
python train_ppo.py --timesteps 500000 --eval-freq 10000
```

**Optimized DQN:**
```bash
python train_optimized.py --episodes 3000 --eval-freq 100
```

### Hyperparameter Search

Run parallel grid search to find optimal hyperparameters:

```bash
python hyperparam_search.py --workers 4 --episodes 1000 --eval-episodes 100
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
├── mining_env.py          # Gymnasium environment
├── minable.json           # Reward definitions
├── game.ipynb             # Interactive notebook demo
│
├── dqn_agent.py           # Dueling Double DQN + PER
├── ppo_agent.py           # PPO with GAE
├── optimized_agent.py     # Enhanced DQN
├── smart_agent.py         # Smart agent with reward shaping
│
├── train_agent.py         # DQN training script
├── train_ppo.py           # PPO training script
├── train_optimized.py     # Optimized DQN training
├── train_smart.py         # Smart agent training
│
├── hyperparam_search.py   # Grid search framework
├── hyperparam_search/     # Search results
└── checkpoints/           # Saved models
```

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
