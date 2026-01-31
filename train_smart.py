#!/usr/bin/env python3
"""
Training script for the Smart Agent with reward shaping.

This agent uses potential-based reward shaping and action masking
to learn effectively in the sparse-reward mining environment.
"""

import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

import mining_env
from smart_agent import SmartAgent, evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train smart agent for mining")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--board-height", type=int, default=10)
    parser.add_argument("--board-width", type=int, default=13)
    parser.add_argument("--max-energy", type=int, default=95)
    parser.add_argument("--minor-rewards", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    # Network architecture
    parser.add_argument("--conv-channels", type=int, default=64,
                        help="Number of channels in conv layers")
    parser.add_argument("--n-conv-layers", type=int, default=3,
                        help="Number of convolutional layers")
    parser.add_argument("--fc-hidden-size", type=int, default=128,
                        help="Size of FC hidden layer in value network")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def train(
    env,
    agent: SmartAgent,
    n_episodes: int,
    eval_freq: int,
    eval_episodes: int,
    save_dir: str
) -> Tuple[List[float], List[Tuple[int, dict]]]:
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    eval_results = []
    best_mean_reward = float('-inf')
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        agent.prev_potential = None
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            
            # Shape reward using potential function
            shaped_reward = agent.shape_reward(reward, state, next_state, done)
            agent.store_transition(state, action, log_prob, shaped_reward, value, done)
            
            episode_reward += reward  # Track raw reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Update at end of episode
        with torch.no_grad():
            dust = torch.tensor(state["dust"]).unsqueeze(0).to(agent.device)
            energy = torch.tensor([state["energy"]]).to(agent.device)
            retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(agent.device)
            next_value = agent.value(dust, energy, retrieved).item()
        
        metrics = agent.update(next_value)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            elapsed = time.time() - start_time
            print(f"Episode {episode + 1:5d} | "
                  f"Avg Reward (100): {avg_reward:6.2f} | "
                  f"Entropy: {metrics.get('entropy', 0):.3f} | "
                  f"Time: {elapsed:.0f}s")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent(agent, env, eval_episodes)
            eval_results.append((episode + 1, eval_metrics))
            print(f"\n{'='*60}")
            print(f"[EVAL] Episode {episode + 1}")
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Mean Retrieved: {eval_metrics['mean_retrieved']:.2f}")
            print(f"  Max Reward: {eval_metrics['max_reward']:.2f}")
            print(f"{'='*60}\n")
            
            if eval_metrics['mean_reward'] > best_mean_reward:
                best_mean_reward = eval_metrics['mean_reward']
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"  ★ NEW BEST: {best_mean_reward:.2f}\n")
    
    agent.save(os.path.join(save_dir, "final_model.pt"))
    return episode_rewards, eval_results


def plot_results(
    episode_rewards: List[float],
    eval_results: List[Tuple[int, dict]],
    random_baseline: float,
    save_dir: str
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.2, color='blue')
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                 color='red', linewidth=2, label='100-Episode Average')
    ax1.axhline(y=random_baseline, color='gray', linestyle='--', label='Random Baseline')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative rewards
    ax2 = axes[0, 1]
    ax2.plot(np.cumsum(episode_rewards), color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards')
    ax2.grid(True, alpha=0.3)
    
    if eval_results:
        episodes = [r[0] for r in eval_results]
        mean_rewards = [r[1]['mean_reward'] for r in eval_results]
        std_rewards = [r[1]['std_reward'] for r in eval_results]
        
        ax3 = axes[1, 0]
        ax3.plot(episodes, mean_rewards, 'b-', marker='o')
        ax3.fill_between(episodes,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.3)
        ax3.axhline(y=random_baseline, color='gray', linestyle='--', label='Random')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Evaluation Reward')
        ax3.set_title('Evaluation Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        mean_retrieved = [r[1]['mean_retrieved'] for r in eval_results]
        ax4 = axes[1, 1]
        ax4.plot(episodes, mean_retrieved, 'g-', marker='s')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Mean Items Retrieved')
        ax4.set_title('Items Retrieved')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(4, max(mean_retrieved) + 0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150)
    plt.show()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Smart Mining Agent Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Board: {args.board_height}x{args.board_width}")
    print(f"Minor Rewards: {args.minor_rewards}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Entropy Coef: {args.entropy_coef}")
    print(f"Network: conv_ch={args.conv_channels}, layers={args.n_conv_layers}, fc={args.fc_hidden_size}")
    print("=" * 60)
    
    env = mining_env.MiningEnv(
        board_shape=(args.board_height, args.board_width),
        max_energy=args.max_energy,
        minor_rewards=args.minor_rewards
    )
    
    device = "cpu" if args.no_cuda else None
    agent = SmartAgent(
        board_shape=(args.board_height, args.board_width),
        n_retrieved=4,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        conv_channels=args.conv_channels,
        n_conv_layers=args.n_conv_layers,
        fc_hidden_size=args.fc_hidden_size,
        device=device
    )
    
    print(f"Device: {agent.device}")
    
    # Random baseline
    print("\nComputing random baseline...")
    random_rewards = []
    for _ in range(100):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            state, reward, done, _, _ = env.step(env.action_space.sample())
            ep_reward += reward
        random_rewards.append(ep_reward)
    random_baseline = np.mean(random_rewards)
    print(f"Random baseline: {random_baseline:.2f} ± {np.std(random_rewards):.2f}")
    print("=" * 60 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"smart_{timestamp}")
    
    episode_rewards, eval_results = train(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_dir=save_dir
    )
    
    print("\n" + "=" * 60)
    print("Final Evaluation (100 episodes)")
    print("=" * 60)
    final_metrics = evaluate_agent(agent, env, n_episodes=100)
    print(f"Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"Mean Retrieved: {final_metrics['mean_retrieved']:.2f}")
    print(f"Max Reward: {final_metrics['max_reward']:.2f}")
    improvement = final_metrics['mean_reward'] - random_baseline
    improvement_pct = 100 * (final_metrics['mean_reward'] / random_baseline - 1)
    print(f"\nImprovement over random: +{improvement:.2f} ({improvement_pct:.0f}%)")
    
    plot_results(episode_rewards, eval_results, random_baseline, save_dir)
    print(f"\nModels saved to {save_dir}")


if __name__ == "__main__":
    main()
