"""
PPO Agent for Grand Underground Mining Game

PPO (Proximal Policy Optimization) is better suited for this environment because:
1. On-policy learning handles sparse rewards better
2. Policy gradient naturally handles large action spaces
3. Entropy bonus maintains exploration
4. Can learn from even zero-reward episodes by improving policy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Optional, Dict


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Actor: Outputs probability distribution over actions
    Critic: Estimates state value V(s)
    """
    
    def __init__(self, board_shape: Tuple[int, int], n_actions: int, n_retrieved: int = 4):
        super().__init__()
        self.board_shape = board_shape
        self.n_actions = n_actions
        h, w = board_shape
        
        # Shared CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        conv_h, conv_w = h // 2, w // 2
        conv_out_size = 64 * conv_h * conv_w
        combined_size = conv_out_size + 1 + n_retrieved
        
        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, dust: torch.Tensor, energy: torch.Tensor, 
                retrieved: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = dust.shape[0]
        
        # Normalize inputs
        dust_norm = dust.unsqueeze(1).float() / 6.0
        energy_norm = energy.float().view(-1, 1) / 100.0
        
        # CNN features
        conv_out = self.conv(dust_norm)
        conv_out = conv_out.view(batch_size, -1)
        
        # Combine features
        combined = torch.cat([conv_out, energy_norm, retrieved.float()], dim=1)
        
        # Shared features
        shared = self.shared(combined)
        
        # Actor and critic outputs
        action_logits = self.actor(shared)
        value = self.critic(shared)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(self, dust, energy, retrieved, action=None):
        """Get action, log probability, entropy, and value."""
        logits, value = self.forward(dust, energy, retrieved)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class RolloutBuffer:
    """Buffer for storing trajectory data during rollout."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO Agent with GAE (Generalized Advantage Estimation).
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int],
        n_retrieved: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_length: int = 2048,
        device: Optional[str] = None
    ):
        self.board_shape = board_shape
        self.h, self.w = board_shape
        self.n_actions = self.h * self.w * 2
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.network = ActorCritic(board_shape, self.n_actions, n_retrieved).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.buffer = RolloutBuffer()
        self.steps = 0
        self.updates = 0
    
    def action_to_tuple(self, action: int) -> Tuple[int, int, int]:
        tool = action % 2
        location = action // 2
        x = location // self.w
        y = location % self.w
        return (x, y, tool)
    
    def tuple_to_action(self, x: int, y: int, tool: int) -> int:
        return (x * self.w + y) * 2 + tool
    
    def _state_to_tensors(self, state: dict):
        dust = torch.tensor(state["dust"]).unsqueeze(0).to(self.device)
        energy = torch.tensor([state["energy"]]).to(self.device)
        retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(self.device)
        return dust, energy, retrieved
    
    def select_action(self, state: dict, training: bool = True) -> Tuple[Tuple[int, int, int], float, float]:
        """Select action and return log prob and value."""
        dust, energy, retrieved = self._state_to_tensors(state)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                dust, energy, retrieved
            )
        
        action_tuple = self.action_to_tuple(action.item())
        return action_tuple, log_prob.item(), value.item()
    
    def store_transition(self, state: dict, action: Tuple[int, int, int],
                        log_prob: float, reward: float, value: float, done: bool):
        """Store transition in rollout buffer."""
        self.buffer.add(state, self.tuple_to_action(*action), log_prob, reward, value, done)
        self.steps += 1
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values + [next_value])
        dones = np.array(self.buffer.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self, next_value: float) -> Dict[str, float]:
        """Perform PPO update."""
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_dust = torch.tensor(
            np.array([s["dust"] for s in self.buffer.states])
        ).to(self.device)
        states_energy = torch.tensor(
            np.array([s["energy"] for s in self.buffer.states])
        ).to(self.device)
        states_retrieved = torch.tensor(
            np.array([s["retrieved"] for s in self.buffer.states])
        ).to(self.device)
        
        actions = torch.tensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs).to(self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Training epochs
        total_loss = 0
        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        n_batches = 0
        
        indices = np.arange(len(self.buffer))
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    states_dust[batch_idx],
                    states_energy[batch_idx],
                    states_retrieved[batch_idx],
                    actions[batch_idx]
                )
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                pg_loss1 = advantages_t[batch_idx] * ratio
                pg_loss2 = advantages_t[batch_idx] * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = F.mse_loss(new_values, returns_t[batch_idx])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = pg_loss + self.value_coef * v_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.mean().item()
                n_batches += 1
        
        self.updates += 1
        self.buffer.clear()
        
        return {
            "loss": total_loss / n_batches,
            "pg_loss": total_pg_loss / n_batches,
            "v_loss": total_v_loss / n_batches,
            "entropy": total_entropy / n_batches,
        }
    
    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.updates = checkpoint["updates"]


def evaluate_agent(agent: PPOAgent, env, n_episodes: int = 100) -> dict:
    """Evaluate agent performance."""
    rewards = []
    retrieved_counts = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
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
