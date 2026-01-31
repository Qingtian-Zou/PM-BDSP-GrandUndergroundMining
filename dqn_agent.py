"""
Improved DQN Agent for Grand Underground Mining Game

This module implements a Dueling Double DQN with Prioritized Experience Replay
for learning optimal mining strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List, Optional


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates state value and action advantage estimation.
    
    This architecture helps the agent learn which states are valuable without
    having to learn the effect of each action at that state.
    """
    
    def __init__(self, board_shape: Tuple[int, int], n_actions: int, n_retrieved: int = 4):
        super().__init__()
        self.board_shape = board_shape
        self.n_actions = n_actions
        h, w = board_shape
        
        # Convolutional feature extractor for the dust grid
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened conv output size
        conv_out_size = 64 * h * w
        
        # Include energy (1) and retrieved status (n_retrieved)
        combined_size = conv_out_size + 1 + n_retrieved
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
        )
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream - estimates A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, dust: torch.Tensor, energy: torch.Tensor, 
                retrieved: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions.
        
        Args:
            dust: (batch, H, W) dust layer grid
            energy: (batch,) remaining energy
            retrieved: (batch, n_retrieved) binary retrieved status
            
        Returns:
            Q-values: (batch, n_actions)
        """
        batch_size = dust.shape[0]
        
        # Normalize inputs
        dust_normalized = dust.unsqueeze(1).float() / 6.0  # Dust values are 0-6
        energy_normalized = energy.float().view(-1, 1) / 100.0  # Normalize energy
        
        # Extract conv features
        conv_features = self.conv(dust_normalized)
        conv_features = conv_features.view(batch_size, -1)
        
        # Combine all features
        combined = torch.cat([
            conv_features,
            energy_normalized,
            retrieved.float()
        ], dim=1)
        
        # Shared features
        shared = self.shared_fc(combined)
        
        # Compute value and advantage
        value = self.value_stream(shared)  # (batch, 1)
        advantage = self.advantage_stream(shared)  # (batch, n_actions)
        
        # Combine using dueling formula: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    
    Allows O(log n) updates and sampling proportional to priorities.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node for given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return total priority."""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """Add new data with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority at given tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """Get data for given cumulative sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using sum tree for efficient sampling.
    
    Transitions with higher TD-error are sampled more frequently, improving
    sample efficiency.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta each sample
            epsilon: Small constant to ensure non-zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
    
    def push(self, state: dict, action: int, reward: float, 
             next_state: dict, done: bool):
        """Add transition with max priority."""
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, experience)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample batch with priority-based probability.
        
        Returns:
            batch: List of transitions
            indices: Tree indices for updating priorities
            weights: Importance sampling weights
        """
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)
        
        segment = self.tree.total() / batch_size
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority
            batch.append(data)
        
        # Compute importance sampling weights
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


class DQNAgent:
    """
    Double DQN agent with Dueling architecture and Prioritized Experience Replay.
    
    This agent learns to select optimal locations and tools for mining
    by estimating action values and following an epsilon-greedy policy.
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int],
        n_retrieved: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: Optional[str] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            board_shape: (height, width) of the game board
            n_retrieved: Number of retrievable rewards
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            device: Device to use ('cuda' or 'cpu')
        """
        self.board_shape = board_shape
        self.h, self.w = board_shape
        self.n_actions = self.h * self.w * 2  # All location-tool combinations
        self.n_retrieved = n_retrieved
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create networks
        self.policy_net = DuelingDQN(board_shape, self.n_actions, n_retrieved).to(self.device)
        self.target_net = DuelingDQN(board_shape, self.n_actions, n_retrieved).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training stats
        self.steps = 0
        self.losses = []
    
    def action_to_tuple(self, action: int) -> Tuple[int, int, int]:
        """Convert flat action index to (x, y, tool) tuple."""
        tool = action % 2
        location = action // 2
        x = location // self.w
        y = location % self.w
        return (x, y, tool)
    
    def tuple_to_action(self, x: int, y: int, tool: int) -> int:
        """Convert (x, y, tool) tuple to flat action index."""
        return (x * self.w + y) * 2 + tool
    
    def select_action(self, state: dict, training: bool = True) -> Tuple[int, int, int]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Game state dictionary
            training: Whether we're training (use epsilon-greedy) or evaluating
            
        Returns:
            (x, y, tool) action tuple
        """
        if training and random.random() < self.epsilon:
            # Random action
            x = random.randint(0, self.h - 1)
            y = random.randint(0, self.w - 1)
            tool = random.randint(0, 1)
            return (x, y, tool)
        
        # Greedy action
        with torch.no_grad():
            dust = torch.tensor(state["dust"]).unsqueeze(0).to(self.device)
            energy = torch.tensor([state["energy"]]).to(self.device)
            retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(dust, energy, retrieved)
            action = q_values.argmax(dim=1).item()
        
        return self.action_to_tuple(action)
    
    def store_transition(self, state: dict, action: Tuple[int, int, int], 
                        reward: float, next_state: dict, done: bool):
        """Store transition in replay buffer."""
        flat_action = self.tuple_to_action(*action)
        self.buffer.push(state, flat_action, reward, next_state, done)
    
    def update_epsilon(self):
        """Decay epsilon linearly."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) 
            * self.steps / self.epsilon_decay_steps
        )
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value or None if buffer too small
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch, indices, weights = self.buffer.sample(self.batch_size)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Convert batch to tensors
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
        
        # Current Q-values
        current_q = self.policy_net(
            states_dust, states_energy, states_retrieved
        ).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use policy net to select actions, target net to evaluate
        with torch.no_grad():
            next_q_policy = self.policy_net(
                next_states_dust, next_states_energy, next_states_retrieved
            )
            next_actions = next_q_policy.argmax(dim=1)
            
            next_q_target = self.target_net(
                next_states_dust, next_states_energy, next_states_retrieved
            ).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q_target
        
        # TD-errors for priority update
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        # Weighted loss
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon
        self.update_epsilon()
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
    
    def save(self, path: str):
        """Save agent state to file."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']


def evaluate_agent(agent: DQNAgent, env, n_episodes: int = 100) -> dict:
    """
    Evaluate agent performance over multiple episodes.
    
    Args:
        agent: Trained DQN agent
        env: Mining environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
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
