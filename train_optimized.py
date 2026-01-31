#!/usr/bin/env python3
"""
Training script for the Optimized DQN Mining Agent.

Key improvements:
- Episode-based epsilon decay
- Better hyperparameters for this environment
- More detailed logging
"""

import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import mining_env
from optimized_agent import OptimizedDQNAgent, evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train optimized DQN agent for mining"
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-freq", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--board-height", type=int, default=10)
    parser.add_argument("--board-width", type=int, default=13)
    parser.add_argument("--max-energy", type=int, default=95)
    parser.add_argument("--minor-rewards", type=float, default=0.5)  # Higher for shaping
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=2000)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def train(
    env,
    agent: OptimizedDQNAgent,
    n_episodes: int,
    eval_freq: int,
    eval_episodes: int,
    save_freq: int,
    save_dir: str
) -> Tuple[List[float], List[Tuple[int, dict]]]:
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    eval_results = []
    best_mean_reward = float('-inf')
    
    start_time = time.time()
    
    # Track additional metrics
    losses = []
    retrieved_history = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_retrieved = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # End of episode
        agent.end_episode()
        episode_rewards.append(episode_reward)
        episode_retrieved = sum(state["retrieved"])
        retrieved_history.append(episode_retrieved)
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_retrieved = np.mean(retrieved_history[-50:])
            avg_loss = np.mean(losses[-1000:]) if losses else 0
            elapsed = time.time() - start_time
            lr = agent.optimizer.param_groups[0]['lr']
            print(f"Ep {episode + 1:5d} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Retrieved: {avg_retrieved:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"LR: {lr:.2e} | "
                  f"Buffer: {len(agent.buffer):5d} | "
                  f"Time: {elapsed:.0f}s")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent(agent, env, eval_episodes)
            eval_results.append((episode + 1, eval_metrics))
            print(f"\n{'='*60}")
            print(f"[EVAL] Episode {episode + 1}")
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Max Reward: {eval_metrics['max_reward']:.2f}")
            print(f"  Mean Retrieved: {eval_metrics['mean_retrieved']:.2f}")
            print(f"{'='*60}\n")
            
            if eval_metrics['mean_reward'] > best_mean_reward:
                best_mean_reward = eval_metrics['mean_reward']
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"  ★ NEW BEST: {best_mean_reward:.2f}\n")
        
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_{episode + 1}.pt"))
    
    agent.save(os.path.join(save_dir, "final_model.pt"))
    
    return episode_rewards, eval_results


def plot_results(
    episode_rewards: List[float],
    eval_results: List[Tuple[int, dict]],
    save_dir: str
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards with moving average
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.2, color='blue')
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                 color='red', linewidth=2, label=f'{window}-Episode Average')
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
    
    # Evaluation metrics
    if eval_results:
        episodes = [r[0] for r in eval_results]
        mean_rewards = [r[1]['mean_reward'] for r in eval_results]
        std_rewards = [r[1]['std_reward'] for r in eval_results]
        
        ax3 = axes[1, 0]
        ax3.plot(episodes, mean_rewards, 'b-', marker='o', markersize=6)
        ax3.fill_between(
            episodes,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.3
        )
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Evaluation Reward')
        ax3.set_title('Evaluation Performance')
        ax3.grid(True, alpha=0.3)
        
        mean_retrieved = [r[1]['mean_retrieved'] for r in eval_results]
        ax4 = axes[1, 1]
        ax4.plot(episodes, mean_retrieved, 'g-', marker='s', markersize=6)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Mean Items Retrieved')
        ax4.set_title('Items Retrieved')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150)
    plt.show()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Optimized DQN Mining Agent Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Board: {args.board_height}x{args.board_width}")
    print(f"Minor Rewards: {args.minor_rewards}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon Decay Episodes: {args.epsilon_decay_episodes}")
    print(f"N-step: {args.n_step}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)
    
    env = mining_env.MiningEnv(
        board_shape=(args.board_height, args.board_width),
        max_energy=args.max_energy,
        minor_rewards=args.minor_rewards
    )
    
    device = "cpu" if args.no_cuda else None
    agent = OptimizedDQNAgent(
        board_shape=(args.board_height, args.board_width),
        n_retrieved=4,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        n_step=args.n_step,
        device=device
    )
    
    print(f"Device: {agent.device}")
    print(f"Action space: {agent.n_actions}")
    print("=" * 60 + "\n")
    
    # Random baseline
    print("Computing random baseline...")
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
    save_dir = os.path.join(args.save_dir, f"optimized_{timestamp}")
    
    episode_rewards, eval_results = train(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        save_dir=save_dir
    )
    
    print("\n" + "=" * 60)
    print("Final Evaluation (100 episodes)")
    print("=" * 60)
    final_metrics = evaluate_agent(agent, env, n_episodes=100)
    print(f"Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"Max Reward: {final_metrics['max_reward']:.2f}")
    print(f"Mean Retrieved: {final_metrics['mean_retrieved']:.2f}")
    print(f"\nImprovement over random: {final_metrics['mean_reward'] - np.mean(random_rewards):.2f}")
    
    plot_results(episode_rewards, eval_results, save_dir)
    print(f"\nModels saved to {save_dir}")


if __name__ == "__main__":
    main()
