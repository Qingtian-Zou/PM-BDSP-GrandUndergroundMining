"""
Smart Agent for Grand Underground Mining Game

This agent uses a fundamentally different approach:
1. Reduces action space by only considering USEFUL cells (those with dust)
2. Uses distance-based reward shaping to guide towards uncovered areas
3. Employs a simpler, more focused policy network
4. Uses potential-based reward shaping for stable learning

Key insight: The environment has 260 actions but most are redundant.
We should only consider cells that are:
- Not fully cleared (dust > 0)
- Near partially uncovered reward areas
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List, Optional, Dict


class SmartPolicyNet(nn.Module):
    """
    Policy network that outputs action probabilities.
    Uses attention over potential action locations.
    
    Args:
        board_shape: (height, width) of the board
        conv_channels: List of channel sizes for conv layers
        n_conv_layers: Number of convolutional layers
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int],
        conv_channels: int = 64,
        n_conv_layers: int = 3
    ):
        super().__init__()
        self.board_shape = board_shape
        h, w = board_shape
        
        # Build CNN dynamically based on parameters
        conv_layers = []
        in_channels = 1
        for i in range(n_conv_layers):
            out_channels = conv_channels if i > 0 else conv_channels // 2
            if i == n_conv_layers - 1:
                out_channels = conv_channels
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Per-cell action scoring
        self.action_conv = nn.Sequential(
            nn.Conv2d(conv_channels + 1, conv_channels // 2, 1),  # +1 for energy channel
            nn.ReLU(),
            nn.Conv2d(conv_channels // 2, 2, 1),  # 2 tools per cell
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, dust: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for each (cell, tool) combination.
        Output shape: (batch, H, W, 2)
        """
        batch_size = dust.shape[0]
        h, w = self.board_shape
        
        # Normalize
        dust_norm = dust.unsqueeze(1).float() / 6.0
        
        # CNN features
        features = self.conv(dust_norm)  # (B, conv_channels, H, W)
        
        # Add energy as a channel
        energy_map = energy.float().view(batch_size, 1, 1, 1).expand(-1, 1, h, w) / 100.0
        combined = torch.cat([features, energy_map], dim=1)
        
        # Per-cell action scores
        action_logits = self.action_conv(combined)  # (B, 2, H, W)
        action_logits = action_logits.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        return action_logits


class ValueNet(nn.Module):
    """
    Value network for state value estimation.
    
    Args:
        board_shape: (height, width) of the board
        n_retrieved: Number of retrieved item slots
        conv_channels: Number of channels in conv layers
        n_conv_layers: Number of convolutional layers
        fc_hidden_size: Size of fully connected hidden layer
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int],
        n_retrieved: int = 4,
        conv_channels: int = 64,
        n_conv_layers: int = 3,
        fc_hidden_size: int = 128
    ):
        super().__init__()
        h, w = board_shape
        
        # Build CNN dynamically
        conv_layers = []
        in_channels = 1
        for i in range(n_conv_layers):
            out_channels = conv_channels if i > 0 else conv_channels // 2
            if i == n_conv_layers - 1:
                out_channels = conv_channels
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
            ])
            in_channels = out_channels
            # Add pooling after first layer
            if i == 0:
                conv_layers.append(nn.MaxPool2d(2))
        
        self.conv = nn.Sequential(*conv_layers)
        
        conv_size = conv_channels * (h // 2) * (w // 2)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_size + 1 + n_retrieved, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, 1)
        )
    
    def forward(self, dust, energy, retrieved):
        batch_size = dust.shape[0]
        
        dust_norm = dust.unsqueeze(1).float() / 6.0
        conv_out = self.conv(dust_norm).view(batch_size, -1)
        
        energy_norm = energy.float().view(-1, 1) / 100.0
        combined = torch.cat([conv_out, energy_norm, retrieved.float()], dim=1)
        
        return self.fc(combined).squeeze(-1)


class SmartAgent:
    """
    Smart agent with shaped rewards and focused action selection.
    
    Args:
        board_shape: (height, width) of the board
        n_retrieved: Number of retrieved item slots
        lr: Learning rate
        gamma: Discount factor
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        conv_channels: Number of channels in conv layers
        n_conv_layers: Number of convolutional layers
        fc_hidden_size: Size of value network's FC hidden layer
        device: Device to use (None for auto-detect)
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int],
        n_retrieved: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        conv_channels: int = 64,
        n_conv_layers: int = 3,
        fc_hidden_size: int = 128,
        device: Optional[str] = None
    ):
        self.board_shape = board_shape
        self.h, self.w = board_shape
        self.n_actions = self.h * self.w * 2
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Store architecture params for saving/loading
        self.conv_channels = conv_channels
        self.n_conv_layers = n_conv_layers
        self.fc_hidden_size = fc_hidden_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.policy = SmartPolicyNet(
            board_shape, conv_channels, n_conv_layers
        ).to(self.device)
        
        self.value = ValueNet(
            board_shape, n_retrieved, conv_channels, n_conv_layers, fc_hidden_size
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        
        # Trajectory storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Tracking
        self.prev_potential = None
    
    def compute_potential(self, state: dict) -> float:
        """
        Potential function for reward shaping.
        Higher potential = closer to uncovering rewards.
        """
        dust = state["dust"]
        retrieved = state["retrieved"]
        energy = state["energy"]
        
        # Low dust = good (closer to uncovering)
        # Use inverse of average dust as potential
        avg_dust = dust.mean()
        dust_potential = (6.0 - avg_dust) / 6.0  # Normalized to [0, 1]
        
        # Count cleared cells
        cleared_ratio = (dust == 0).sum() / dust.size
        
        # Combine potentials
        potential = 5.0 * dust_potential + 10.0 * cleared_ratio
        
        return potential
    
    def shape_reward(self, reward: float, state: dict, next_state: dict, done: bool) -> float:
        """
        Apply potential-based reward shaping.
        F(s, s') = gamma * phi(s') - phi(s)
        """
        if done:
            self.prev_potential = None
            return reward
        
        current_potential = self.compute_potential(next_state)
        
        if self.prev_potential is None:
            self.prev_potential = self.compute_potential(state)
        
        shaping = self.gamma * current_potential - self.prev_potential
        self.prev_potential = current_potential
        
        return reward + shaping
    
    def action_to_tuple(self, action: int) -> Tuple[int, int, int]:
        tool = action % 2
        location = action // 2
        x = location // self.w
        y = location % self.w
        return (x, y, tool)
    
    def select_action(self, state: dict, training: bool = True) -> Tuple[Tuple[int, int, int], float, float]:
        dust = torch.tensor(state["dust"]).unsqueeze(0).to(self.device)
        energy = torch.tensor([state["energy"]]).to(self.device)
        retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(dust, energy)  # (1, H, W, 2)
            value = self.value(dust, energy, retrieved)
        
        # Flatten logits
        flat_logits = logits.reshape(1, -1)  # (1, H*W*2)
        
        # Create mask for valid actions (cells with dust > 0)
        dust_np = state["dust"]
        valid_mask = (dust_np > 0).flatten()  # (H*W,)
        valid_mask = np.repeat(valid_mask, 2)  # (H*W*2,) for both tools
        valid_mask = torch.tensor(valid_mask).unsqueeze(0).to(self.device)
        
        # Apply mask (set invalid actions to -inf)
        masked_logits = flat_logits.clone()
        masked_logits[~valid_mask] = float('-inf')
        
        # If all masked (shouldn't happen), use all actions
        if valid_mask.sum() == 0:
            masked_logits = flat_logits
        
        probs = F.softmax(masked_logits, dim=-1)
        
        if training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        else:
            action = probs.argmax(dim=-1).item()
            log_prob = 0.0
        
        return self.action_to_tuple(action), log_prob, value.item()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        action_idx = (action[0] * self.w + action[1]) * 2 + action[2]
        self.actions.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self, next_value: float) -> Dict[str, float]:
        """Update policy using collected trajectory."""
        if len(self.states) == 0:
            return {}
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = next_value
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                R = 0
            R = self.rewards[i] + self.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - self.values[i])
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert states to tensors
        dust = torch.tensor(np.array([s["dust"] for s in self.states])).to(self.device)
        energy = torch.tensor(np.array([s["energy"] for s in self.states])).to(self.device)
        retrieved = torch.tensor(np.array([s["retrieved"] for s in self.states])).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        old_log_probs = torch.tensor(self.log_probs).to(self.device)
        
        # Forward pass
        logits = self.policy(dust, energy).reshape(len(self.states), -1)
        values = self.value(dust, energy, retrieved)
        
        # Policy loss
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Policy gradient loss
        pg_loss = -(action_log_probs * advantages).mean()
        
        # Value loss
        v_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()), 0.5
        )
        self.optimizer.step()
        
        # Clear trajectory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy.item(),
        }
    
    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "conv_channels": self.conv_channels,
                "n_conv_layers": self.n_conv_layers,
                "fc_hidden_size": self.fc_hidden_size,
            }
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def evaluate_agent(agent, env, n_episodes: int = 100) -> dict:
    rewards = []
    retrieved_counts = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        agent.prev_potential = None
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, training=False)
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
