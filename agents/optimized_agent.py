"""
Optimized DQN Agent for Grand Underground Mining Game

Key optimizations over the original:
1. Soft target network updates (τ=0.005) for stability
2. Episode-based epsilon decay (not step-based)
3. Lower gamma (0.95) for shorter planning horizon - better for sparse rewards
4. Reward normalization during training
5. Larger batch size and more frequent training
6. Better neural network architecture with dropout
7. N-step returns for faster learning
8. Intrinsic curiosity bonus for exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List, Optional, Dict


class OptimizedDQN(nn.Module):
    """
    Optimized Dueling DQN with better architecture for the mining game.
    """
    
    def __init__(self, board_shape: Tuple[int, int], n_actions: int, n_retrieved: int = 4):
        super().__init__()
        self.board_shape = board_shape
        self.n_actions = n_actions
        h, w = board_shape
        
        # Simpler but effective CNN - avoid overfitting
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial size
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened conv output size after pooling
        conv_h, conv_w = h // 2, w // 2
        conv_out_size = 64 * conv_h * conv_w
        
        # Include energy (1) and retrieved status (n_retrieved)
        combined_size = conv_out_size + 1 + n_retrieved
        
        # Shared feature layer with dropout
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, dust: torch.Tensor, energy: torch.Tensor, 
                retrieved: torch.Tensor) -> torch.Tensor:
        batch_size = dust.shape[0]
        
        # Normalize inputs
        dust_normalized = dust.unsqueeze(1).float() / 6.0
        energy_normalized = energy.float().view(-1, 1) / 100.0
        
        # CNN features
        conv_features = self.conv(dust_normalized)
        conv_features = conv_features.view(batch_size, -1)
        
        # Combine features
        combined = torch.cat([
            conv_features,
            energy_normalized,
            retrieved.float()
        ], dim=1)
        
        # Forward through streams
        shared = self.shared_fc(combined)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Dueling: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ReplayBuffer:
    """
    Simple but efficient replay buffer with N-step returns.
    """
    
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.95):
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def _get_n_step_info(self):
        """Compute N-step return and final state."""
        reward, next_state, done = self.n_step_buffer[-1][2:]
        
        # Accumulate discounted rewards from n-step buffer
        for i in range(len(self.n_step_buffer) - 2, -1, -1):
            r, ns, d = self.n_step_buffer[i][2:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = ns, d
        
        return reward, next_state, done
    
    def push(self, state: dict, action: int, reward: float, 
             next_state: dict, done: bool):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Only store when we have enough steps or episode ends
        if len(self.n_step_buffer) == self.n_step or done:
            reward_n, next_state_n, done_n = self._get_n_step_info()
            state_0, action_0 = self.n_step_buffer[0][:2]
            self.buffer.append((state_0, action_0, reward_n, next_state_n, done_n))
        
        if done:
            # Flush remaining transitions
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                if self.n_step_buffer:
                    reward_n, next_state_n, done_n = self._get_n_step_info()
                    state_0, action_0 = self.n_step_buffer[0][:2]
                    self.buffer.append((state_0, action_0, reward_n, next_state_n, done_n))
            self.n_step_buffer.clear()
    
    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class OptimizedDQNAgent:
    """
    Optimized DQN agent with key improvements for stable learning.
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int],
        n_retrieved: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.95,  # Lower gamma for sparse rewards
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 3000,  # Episode-based decay
        buffer_size: int = 50000,
        batch_size: int = 128,  # Larger batch
        tau: float = 0.005,  # Soft update coefficient
        n_step: int = 3,  # N-step returns
        train_freq: int = 4,  # Train every N steps
        device: Optional[str] = None
    ):
        self.board_shape = board_shape
        self.h, self.w = board_shape
        self.n_actions = self.h * self.w * 2
        self.n_retrieved = n_retrieved
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.batch_size = batch_size
        self.tau = tau
        self.train_freq = train_freq
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = OptimizedDQN(board_shape, self.n_actions, n_retrieved).to(self.device)
        self.target_net = OptimizedDQN(board_shape, self.n_actions, n_retrieved).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, n_step=n_step, gamma=gamma)
        
        # Stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # State visitation for curiosity
        self.state_counts = {}
    
    def action_to_tuple(self, action: int) -> Tuple[int, int, int]:
        tool = action % 2
        location = action // 2
        x = location // self.w
        y = location % self.w
        return (x, y, tool)
    
    def tuple_to_action(self, x: int, y: int, tool: int) -> int:
        return (x * self.w + y) * 2 + tool
    
    def _get_state_hash(self, state: dict) -> int:
        """Hash state for curiosity bonus."""
        # Simple hash based on dust pattern
        dust = state["dust"]
        # Discretize to reduce state space
        discretized = (dust > 3).astype(np.int8)
        return hash(discretized.tobytes())
    
    def _get_curiosity_bonus(self, state: dict) -> float:
        """Intrinsic reward for visiting new states."""
        state_hash = self._get_state_hash(state)
        count = self.state_counts.get(state_hash, 0)
        self.state_counts[state_hash] = count + 1
        
        # Curiosity bonus: 1/sqrt(count + 1)
        return 0.1 / np.sqrt(count + 1)
    
    def select_action(self, state: dict, training: bool = True) -> Tuple[int, int, int]:
        if training and random.random() < self.epsilon:
            # Smart random: prefer low dust areas
            dust = state["dust"]
            if random.random() < 0.3:  # 30% completely random
                x = random.randint(0, self.h - 1)
                y = random.randint(0, self.w - 1)
            else:
                # Prefer areas with lower dust
                flat_dust = dust.flatten()
                # Inverse probability - lower dust = higher probability
                probs = np.exp(-flat_dust / 2.0)
                probs = probs / probs.sum()
                flat_idx = np.random.choice(len(flat_dust), p=probs)
                x = flat_idx // self.w
                y = flat_idx % self.w
            tool = random.randint(0, 1)
            return (x, y, tool)
        
        with torch.no_grad():
            dust = torch.tensor(state["dust"]).unsqueeze(0).to(self.device)
            energy = torch.tensor([state["energy"]]).to(self.device)
            retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(dust, energy, retrieved)
            action = q_values.argmax(dim=1).item()
        
        return self.action_to_tuple(action)
    
    def store_transition(self, state: dict, action: Tuple[int, int, int], 
                        reward: float, next_state: dict, done: bool):
        flat_action = self.tuple_to_action(*action)
        
        # Add curiosity bonus
        curiosity = self._get_curiosity_bonus(next_state)
        augmented_reward = reward + curiosity
        
        self.buffer.push(state, flat_action, augmented_reward, next_state, done)
        self.steps += 1
        
        # Update reward statistics
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(
            (self.reward_std ** 2 * (self.reward_count - 1) + delta * delta2) / self.reward_count
        )
    
    def update_epsilon(self):
        """Episode-based epsilon decay."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) 
            * self.episodes / self.epsilon_decay_episodes
        )
    
    def end_episode(self):
        """Call at end of each episode."""
        self.episodes += 1
        self.update_epsilon()
        self.scheduler.step()
    
    def _soft_update_target(self):
        """Soft update target network: θ' = τ*θ + (1-τ)*θ'"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        
        if self.steps % self.train_freq != 0:
            return None
        
        batch = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_dust = torch.tensor(
            np.array([t[0]["dust"] for t in batch])
        ).to(self.device)
        states_energy = torch.tensor(
            np.array([t[0]["energy"] for t in batch])
        ).to(self.device)
        states_retrieved = torch.tensor(
            np.array([t[0]["retrieved"] for t in batch])
        ).to(self.device)
        
        actions = torch.tensor([t[1] for t in batch]).to(self.device)
        rewards = torch.tensor(
            [t[2] for t in batch], dtype=torch.float32
        ).to(self.device)
        
        next_states_dust = torch.tensor(
            np.array([t[3]["dust"] for t in batch])
        ).to(self.device)
        next_states_energy = torch.tensor(
            np.array([t[3]["energy"] for t in batch])
        ).to(self.device)
        next_states_retrieved = torch.tensor(
            np.array([t[3]["retrieved"] for t in batch])
        ).to(self.device)
        
        dones = torch.tensor(
            [t[4] for t in batch], dtype=torch.float32
        ).to(self.device)
        
        # Normalize rewards
        if self.reward_std > 0:
            rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        rewards = torch.clamp(rewards, -10, 10)
        
        # Current Q-values
        current_q = self.policy_net(
            states_dust, states_energy, states_retrieved
        ).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_q_policy = self.policy_net(
                next_states_dust, next_states_energy, next_states_retrieved
            )
            next_actions = next_q_policy.argmax(dim=1)
            
            next_q_target = self.target_net(
                next_states_dust, next_states_energy, next_states_retrieved
            ).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # N-step already accounted for in buffer
            gamma_n = self.gamma ** self.buffer.n_step
            target_q = rewards + (1 - dones) * gamma_n * next_q_target
        
        # Huber loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target
        self._soft_update_target()
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
    
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        if 'reward_mean' in checkpoint:
            self.reward_mean = checkpoint['reward_mean']
            self.reward_std = checkpoint['reward_std']
            self.reward_count = checkpoint['reward_count']


def evaluate_agent(agent, env, n_episodes: int = 100) -> dict:
    """Evaluate agent performance."""
    rewards = []
    retrieved_counts = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        retrieved_counts.append(sum(state["retrieved"]))
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
        "mean_retrieved": np.mean(retrieved_counts),
    }
