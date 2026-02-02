#!/usr/bin/env python3
"""
Training script for PPO Mining Agent.

PPO typically works better than DQN for:
- Large discrete action spaces
- Sparse rewards
- Need for stable exploration
"""

import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game import MiningEnv
from agents.ppo_agent import PPOAgent, evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for mining")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--board-height", type=int, default=10)
    parser.add_argument("--board-width", type=int, default=13)
    parser.add_argument("--max-energy", type=int, default=95)
    parser.add_argument("--minor-rewards", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def train(
    env,
    agent: PPOAgent,
    total_timesteps: int,
    eval_freq: int,
    eval_episodes: int,
    save_dir: str
) -> Tuple[List[float], List[Tuple[int, dict]]]:
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    eval_results = []
    best_mean_reward = float('-inf')
    
    start_time = time.time()
    
    current_episode_reward = 0
    n_episodes = 0
    state, _ = env.reset()
    
    update_metrics = []
    
    for step in range(total_timesteps):
        # Collect action
        action, log_prob, value = agent.select_action(state, training=True)
        
        # Step environment
        next_state, reward, done, _, _ = env.step(action)
        current_episode_reward += reward
        
        # Store transition
        agent.store_transition(state, action, log_prob, reward, value, done)
        state = next_state
        
        if done:
            episode_rewards.append(current_episode_reward)
            n_episodes += 1
            current_episode_reward = 0
            state, _ = env.reset()
        
        # Update when buffer is full
        if len(agent.buffer) >= agent.rollout_length:
            # Get value for last state
            with torch.no_grad():
                dust = torch.tensor(state["dust"]).unsqueeze(0).to(agent.device)
                energy = torch.tensor([state["energy"]]).to(agent.device)
                retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(agent.device)
                _, next_value = agent.network(dust, energy, retrieved)
            
            metrics = agent.update(next_value.item())
            update_metrics.append(metrics)
            
            # Log
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                elapsed = time.time() - start_time
                fps = step / elapsed if elapsed > 0 else 0
                print(f"Step {step:7d} | "
                      f"Episodes: {n_episodes:4d} | "
                      f"Avg Reward (10): {avg_reward:6.2f} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Entropy: {metrics['entropy']:.3f} | "
                      f"FPS: {fps:.0f}")
        
        # Evaluation
        if (step + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent(agent, env, eval_episodes)
            eval_results.append((step + 1, eval_metrics))
            print(f"\n{'='*60}")
            print(f"[EVAL] Step {step + 1}")
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Mean Retrieved: {eval_metrics['mean_retrieved']:.2f}")
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
    save_dir: str
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.2, color='blue')
    window = min(100, len(episode_rewards) // 10 + 1)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                 color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(np.cumsum(episode_rewards), color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards')
    ax2.grid(True, alpha=0.3)
    
    if eval_results:
        steps = [r[0] for r in eval_results]
        mean_rewards = [r[1]['mean_reward'] for r in eval_results]
        std_rewards = [r[1]['std_reward'] for r in eval_results]
        
        ax3 = axes[1, 0]
        ax3.plot(steps, mean_rewards, 'b-', marker='o')
        ax3.fill_between(steps,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.3)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Evaluation Reward')
        ax3.set_title('Evaluation Performance')
        ax3.grid(True, alpha=0.3)
        
        mean_retrieved = [r[1]['mean_retrieved'] for r in eval_results]
        ax4 = axes[1, 1]
        ax4.plot(steps, mean_retrieved, 'g-', marker='s')
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Mean Items Retrieved')
        ax4.set_title('Items Retrieved')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150)
    plt.show()


# Need torch import for main
import torch

def main():
    args = parse_args()
    
    print("=" * 60)
    print("PPO Mining Agent Training")
    print("=" * 60)
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Board: {args.board_height}x{args.board_width}")
    print(f"Minor Rewards: {args.minor_rewards}")
    print(f"Learning Rate: {args.lr}")
    print(f"Rollout Length: {args.rollout_length}")
    print(f"Entropy Coef: {args.entropy_coef}")
    print("=" * 60)
    
    env = MiningEnv(
        board_shape=(args.board_height, args.board_width),
        max_energy=args.max_energy,
        minor_rewards=args.minor_rewards
    )
    
    device = "cpu" if args.no_cuda else None
    agent = PPOAgent(
        board_shape=(args.board_height, args.board_width),
        n_retrieved=4,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        rollout_length=args.rollout_length,
        device=device
    )
    
    print(f"Device: {agent.device}")
    print(f"Action space: {agent.n_actions}")
    
    # Random baseline
    print("\nComputing random baseline...")
    random_rewards = []
    for _ in range(100):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        random_rewards.append(ep_reward)
    print(f"Random baseline: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print("=" * 60 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"ppo_{timestamp}")
    
    episode_rewards, eval_results = train(
        env=env,
        agent=agent,
        total_timesteps=args.total_timesteps,
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
    print(f"\nImprovement over random: {final_metrics['mean_reward'] - np.mean(random_rewards):.2f}")
    
    plot_results(episode_rewards, eval_results, save_dir)
    print(f"\nModels saved to {save_dir}")


if __name__ == "__main__":
    main()
