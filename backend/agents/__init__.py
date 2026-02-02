"""
Agent implementations for the Grand Underground Mining Game.

This package contains various RL agents:
- DQNAgent: Dueling Double DQN with Prioritized Experience Replay
- OptimizedDQNAgent: Optimized DQN with soft target updates and N-step returns
- PPOAgent: Proximal Policy Optimization agent
- SmartAgent: Policy gradient agent with reward shaping
"""

from agents.dqn_agent import DQNAgent, DuelingDQN, PrioritizedReplayBuffer
from agents.optimized_agent import OptimizedDQNAgent, OptimizedDQN, ReplayBuffer
from agents.ppo_agent import PPOAgent, ActorCritic, RolloutBuffer
from agents.smart_agent import SmartAgent, SmartPolicyNet, ValueNet

__all__ = [
    'DQNAgent', 'DuelingDQN', 'PrioritizedReplayBuffer',
    'OptimizedDQNAgent', 'OptimizedDQN', 'ReplayBuffer',
    'PPOAgent', 'ActorCritic', 'RolloutBuffer',
    'SmartAgent', 'SmartPolicyNet', 'ValueNet',
]
