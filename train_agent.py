#!/usr/bin/env python3
"""
Training script for the DQN Mining Agent.

This script trains a Dueling Double DQN agent to play the Grand Underground
mining minigame from Pokemon BDSP.

Usage:
    python train_agent.py [--episodes N] [--eval-freq N] [--save-freq N]
"""

import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import mining_env
from dqn_agent import DQNAgent, evaluate_agent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for the mining minigame"
    )
    parser.add_argument(
        "--episodes", type=int, default=10000,
        help="Number of training episodes (default: 10000)"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=500,
        help="Evaluate every N episodes (default: 500)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50,
        help="Number of episodes for evaluation (default: 50)"
    )
    parser.add_argument(
        "--save-freq", type=int, default=1000,
        help="Save checkpoint every N episodes (default: 1000)"
    )
    parser.add_argument(
        "--board-height", type=int, default=10,
        help="Board height (default: 10)"
    )
    parser.add_argument(
        "--board-width", type=int, default=13,
        help="Board width (default: 13)"
    )
    parser.add_argument(
        "--max-energy", type=int, default=95,
        help="Maximum energy (default: 95)"
    )
    parser.add_argument(
        "--minor-rewards", type=float, default=0.1,
        help="Minor reward for partial uncovering (default: 0.1)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=8000,
        help="Steps to decay epsilon (default: 8000)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000,
        help="Replay buffer size (default: 100000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--target-update-freq", type=int, default=500,
        help="Target network update frequency (default: 500)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Disable CUDA training"
    )
    return parser.parse_args()


def train(
    env,
    agent: DQNAgent,
    n_episodes: int,
    eval_freq: int,
    eval_episodes: int,
    save_freq: int,
    save_dir: str
) -> Tuple[List[float], List[Tuple[int, dict]]]:
    """
    Train the agent.
    
    Args:
        env: Mining environment
        agent: DQN agent
        n_episodes: Total training episodes
        eval_freq: Evaluate every N episodes
        eval_episodes: Episodes per evaluation
        save_freq: Save checkpoint every N episodes
        save_dir: Directory for checkpoints
        
    Returns:
        episode_rewards: List of episode rewards
        eval_results: List of (episode, metrics) tuples
    """
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    eval_results = []
    best_mean_reward = float('-inf')
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            elapsed = time.time() - start_time
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer)} | "
                  f"Time: {elapsed:.1f}s")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent(agent, env, eval_episodes)
            eval_results.append((episode + 1, eval_metrics))
            print(f"\n[EVAL] Episode {episode + 1}")
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Max Reward: {eval_metrics['max_reward']:.2f}")
            print(f"  Mean Retrieved: {eval_metrics['mean_retrieved']:.2f}\n")
            
            # Save best model
            if eval_metrics['mean_reward'] > best_mean_reward:
                best_mean_reward = eval_metrics['mean_reward']
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"  [NEW BEST] Saved best model with reward {best_mean_reward:.2f}\n")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_{episode + 1}.pt"))
    
    # Save final model
    agent.save(os.path.join(save_dir, "final_model.pt"))
    
    return episode_rewards, eval_results


def plot_results(
    episode_rewards: List[float],
    eval_results: List[Tuple[int, dict]],
    save_dir: str
):
    """Generate and save training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                 color='red', label=f'{window}-Episode Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative rewards
    ax2 = axes[0, 1]
    ax2.plot(np.cumsum(episode_rewards))
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards')
    ax2.grid(True, alpha=0.3)
    
    # Evaluation metrics
    if eval_results:
        episodes = [r[0] for r in eval_results]
        mean_rewards = [r[1]['mean_reward'] for r in eval_results]
        std_rewards = [r[1]['std_reward'] for r in eval_results]
        
        ax3 = axes[1, 0]
        ax3.plot(episodes, mean_rewards, 'b-', marker='o', label='Mean Reward')
        ax3.fill_between(
            episodes,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.3
        )
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Evaluation Reward')
        ax3.set_title('Evaluation Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Retrieved items
        mean_retrieved = [r[1]['mean_retrieved'] for r in eval_results]
        ax4 = axes[1, 1]
        ax4.plot(episodes, mean_retrieved, 'g-', marker='s')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Mean Items Retrieved')
        ax4.set_title('Items Retrieved Over Training')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150)
    plt.show()
    print(f"Saved training plots to {os.path.join(save_dir, 'training_results.png')}")


def main():
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("DQN Mining Agent Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Board: {args.board_height}x{args.board_width}")
    print(f"Max Energy: {args.max_energy}")
    print(f"Minor Rewards: {args.minor_rewards}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon Decay Steps: {args.epsilon_decay_steps}")
    print(f"Buffer Size: {args.buffer_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Target Update Freq: {args.target_update_freq}")
    print("=" * 60)
    
    # Create environment
    env = mining_env.MiningEnv(
        board_shape=(args.board_height, args.board_width),
        max_energy=args.max_energy,
        minor_rewards=args.minor_rewards
    )
    
    # Create agent
    device = "cpu" if args.no_cuda else None
    agent = DQNAgent(
        board_shape=(args.board_height, args.board_width),
        n_retrieved=4,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay_steps=args.epsilon_decay_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=device
    )
    
    print(f"Device: {agent.device}")
    print(f"Action space size: {agent.n_actions}")
    print("=" * 60 + "\n")
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    
    # Train
    episode_rewards, eval_results = train(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        save_dir=save_dir
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (100 episodes)")
    print("=" * 60)
    final_metrics = evaluate_agent(agent, env, n_episodes=100)
    print(f"Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"Max Reward: {final_metrics['max_reward']:.2f}")
    print(f"Min Reward: {final_metrics['min_reward']:.2f}")
    print(f"Mean Retrieved: {final_metrics['mean_retrieved']:.2f}")
    
    # Plot results
    plot_results(episode_rewards, eval_results, save_dir)
    
    print(f"\nTraining complete! Models saved to {save_dir}")


if __name__ == "__main__":
    main()
