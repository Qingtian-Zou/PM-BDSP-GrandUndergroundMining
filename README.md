# PM-BDSP-GrandUndergroundMining

A reinforcement learning environment and agent framework for simulating the mining minigame from Pokémon Brilliant Diamond & Shining Pearl's Grand Underground. This project provides a customizable Gymnasium-compatible environment, reward definitions, and deep RL agent examples for research and experimentation.

---

## Features

- **Custom RL Environment:**  
  - Rectangular grid board with dust layers and randomly placed, non-overlapping rewards.
  - Reward shapes, sizes, and values are defined in [`minable.json`](minable.json).
  - Two mining tools: brush and blower, each with unique dust-clearing patterns and energy costs.
  - Partial (minor) rewards for uncovering parts of a reward, and major rewards for fully retrieving them.
  - Episode ends when energy runs out or all rewards are retrieved.

- **Deep RL Agent Examples:**  
  - Two-stage agent: first selects a location, then chooses a tool.
  - PyTorch neural network architectures for both location and tool selection.
  - Experience replay, target networks, and epsilon-greedy exploration.
  - Training loop and reward visualization in Jupyter notebook.

---

## Getting Started

### Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `torch`
- `matplotlib`

Install dependencies:
```bash
pip install gymnasium numpy torch matplotlib
```

### Environment Usage

```python
import mining_env

env = mining_env.MiningEnv(board_shape=(12, 16), max_energy=120, minor_rewards=0.01)
obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print("Reward:", reward)
```

### RL Agent Training

See [`game.ipynb`](game.ipynb) for an RL agent implementation and training loop demonstration.  
The agent uses two neural networks: one for selecting the mining location, and one for selecting the tool.

---

## Reward Definitions

All possible rewards, their shapes, and values are defined in [`minable.json`](minable.json).  
You can add or modify rewards by editing this file.

Example entry:
```json
"Sphere S": {
    "size": 4,
    "shape": [[1,1],[1,1]],
    "color": "white",
    "value": 4
}
```

---

## Environment Details

- **Action Space:**  
  `(x, y, tool)` where `x` and `y` are grid coordinates, and `tool` is 0 (brush) or 1 (blower).
- **Observation Space:**  
  - `dust`: 2D array of dust layers.
  - `energy`: remaining energy.
  - `retrieved`: binary vector indicating which rewards have been collected.

- **Tools:**
  - **Brush:** Clears 2 dust on center, 1 on 4-neighbors. Costs 8 energy.
  - **Blower:** Clears 3 on center, 2 on 8-neighbors, 1 on outer ring. Costs 20 energy.

- **Rewards:**
  - **Major:** For fully uncovering a reward.
  - **Minor:** For each new reward-tile grid uncovered in a step (`minor_rewards` parameter).

---

## Visualization

After training, cumulative rewards can be visualized using matplotlib:

```python
import matplotlib.pyplot as plt
plt.plot(np.cumsum(history))
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training History")
plt.show()
```

---

## Tips for Better Learning

- Use reward shaping (minor rewards) to encourage exploration.
- Normalize inputs for neural networks.
- Adjust network size, learning rate, and exploration schedule for best results.
- See comments in [`game.ipynb`](game.ipynb) for more ideas.

---

## License

MIT License

---

## Acknowledgements

Inspired by the mining minigame in Pokémon Brilliant Diamond & Shining Pearl.
