import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
from pathlib import Path

class MiningEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_shape=(10, 10), max_energy=100, minable_path="minable.json", minor_rewards=0.0):
        super().__init__()
        self.board_shape = board_shape
        self.max_energy = max_energy
        self.minable_path = minable_path
        self.minor_rewards = minor_rewards # give minor rewards for partial reward block revealing
        self.action_space = spaces.MultiDiscrete([board_shape[0], board_shape[1], 2])  # (x, y, tool)
        self.observation_space = spaces.Dict({
            "dust": spaces.Box(0, 5, shape=board_shape, dtype=np.int8),
            "energy": spaces.Discrete(max_energy + 1),
            "retrieved": spaces.MultiBinary(5),  # max 5 rewards
        })
        self._load_minables()
        self.reset()

    def _load_minables(self):
        with open(self.minable_path) as f:
            self.minables = list(json.load(f).items())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.energy = self.max_energy
        self.dust = np.zeros(self.board_shape, dtype=np.int8)
        self.rewards = []
        self.reward_masks = []
        self.reward_values = []
        self.retrieved = np.zeros(5, dtype=np.int8)
        self._place_rewards()
        self._spread_dust()
        obs = self._get_obs()
        return obs, {}

    def _place_rewards(self):
        n_rewards = random.randint(2, 5)
        chosen = random.sample(self.minables, n_rewards)
        board = np.zeros(self.board_shape, dtype=np.int8)
        self.rewards = []
        self.reward_masks = []
        self.reward_values = []
        for name, info in chosen:
            shape = np.array(info["shape"], dtype=np.int8)
            sh = shape.shape
            tries = 0
            while True:
                x = random.randint(0, self.board_shape[0] - sh[0])
                y = random.randint(0, self.board_shape[1] - sh[1])
                region = board[x:x+sh[0], y:y+sh[1]]
                if np.all(region == 0):
                    board[x:x+sh[0], y:y+sh[1]] += shape
                    self.rewards.append((name, (x, y)))
                    mask = np.zeros(self.board_shape, dtype=bool)
                    mask[x:x+sh[0], y:y+sh[1]] = shape.astype(bool)
                    self.reward_masks.append(mask)
                    self.reward_values.append(info["value"])
                    break
                tries += 1
                if tries > 1000:
                    raise RuntimeError("Failed to place rewards without overlap.")
        self.reward_board = board

    def _spread_dust(self):
        min_layer = 2
        max_layer = 5
        dust = np.zeros(self.board_shape, dtype=np.int8)
        # Start with a random seed point
        x, y = random.randint(0, self.board_shape[0]-1), random.randint(0, self.board_shape[1]-1)
        dust[x, y] = random.randint(min_layer, max_layer)
        queue = [(x, y)]
        visited = set(queue)
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.board_shape[0] and 0 <= ny < self.board_shape[1]:
                    if (nx, ny) not in visited:
                        # Set dust layer to be within 1 of neighbors
                        neighbor_layers = [dust[cx, cy]]
                        for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nnx, nny = nx+ddx, ny+ddy
                            if 0 <= nnx < self.board_shape[0] and 0 <= nny < self.board_shape[1]:
                                neighbor_layers.append(dust[nnx, nny])
                        min_n = max(min(neighbor_layers)-1, min_layer)
                        max_n = min(max(neighbor_layers)+1, max_layer)
                        dust[nx, ny] = random.randint(min_n, max_n)
                        queue.append((nx, ny))
                        visited.add((nx, ny))
        self.dust = dust

    def _get_obs(self):
        return {
            "dust": self.dust.copy(),
            "energy": self.energy,
            "retrieved": self.retrieved.copy(),
        }

    def step(self, action):
        x, y, tool = action
        if self.energy <= 0:
            return self._get_obs(), 0, True, False, {}
        if tool == 0:
            # Brush: 2 on center, 1 on 4-neighbors
            self.energy -= 8
            self.dust[x, y] = max(0, self.dust[x, y] - 2)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.board_shape[0] and 0 <= ny < self.board_shape[1]:
                    self.dust[nx, ny] = max(0, self.dust[nx, ny] - 1)
        else:
            # Blower: 3 on center, 2 on 8-neighbors, 1 on outer ring
            self.energy -= 20
            self.dust[x, y] = max(0, self.dust[x, y] - 3)
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.board_shape[0] and 0 <= ny < self.board_shape[1]:
                        self.dust[nx, ny] = max(0, self.dust[nx, ny] - 2)
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.board_shape[0] and 0 <= ny < self.board_shape[1]:
                    self.dust[nx, ny] = max(0, self.dust[nx, ny] - 1)
        # Check for retrieved rewards
        reward = 0
        for i, mask in enumerate(self.reward_masks):
            if self.retrieved[i]:
                continue
            # Minor reward for each newly revealed reward block (partial uncovering)
            if self.minor_rewards > 0:
                # Count how many reward blocks were just revealed this step
                prev_uncovered = np.sum((self.dust[mask] + (self.dust[mask] == 0)) == 0)  # all blocks covered last step
                now_uncovered = np.sum(self.dust[mask] == 0)
                newly_uncovered = now_uncovered - prev_uncovered
                if newly_uncovered > 0:
                    reward += self.minor_rewards * newly_uncovered
            if np.all(self.dust[mask] == 0):
                self.retrieved[i] = 1
                reward += self.reward_values[i]
        done = (self.energy <= 0) or np.all(self.retrieved)
        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        print("Energy:", self.energy)
        print("Dust:")
        print(self.dust)
        print("Retrieved:", self.retrieved)
        # Show rewards on board
        reward_board = np.full(self.board_shape, ".", dtype=object)
        for idx, ((name, (x, y)), mask) in enumerate(zip(self.rewards, self.reward_masks)):
            symbol = str(idx+1) if not self.retrieved[idx] else "*"
            reward_board[mask] = symbol
        print("Rewards on board (number = unretrieved, * = retrieved):")
        for row in reward_board:
            print(" ".join(row))