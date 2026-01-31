#!/usr/bin/env python3
"""
Hyperparameter Grid Search for Smart Mining Agent

Runs parallel experiments across different hyperparameter combinations
and reports the best configuration.

Usage:
    python hyperparam_search.py --workers 4 --episodes 1000
    
    # With network architecture search
    python hyperparam_search.py --workers 4 --episodes 1000 \
        --conv-channels 32 64 128 --n-conv-layers 2 3 4
"""

import argparse
import csv
import os
import json
import time
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Any
import numpy as np

import mining_env
from smart_agent import SmartAgent, evaluate_agent
import torch


# Define hyperparameter search space
PARAM_GRID = {
    # Learning hyperparameters
    "lr": [5e-4, 1e-3, 2e-3],
    "gamma": [0.9, 0.95, 0.99],
    "entropy_coef": [0.01, 0.05, 0.1],
    "minor_rewards": [0.3, 0.5, 0.7],
    # Network architecture
    "conv_channels": [32, 64, 128],
    "n_conv_layers": [2, 3, 4],
    "fc_hidden_size": [64, 128, 256],
}


def train_single_config(args: Tuple[int, Dict[str, Any], int, int]) -> Dict[str, Any]:
    """
    Train agent with a single hyperparameter configuration.
    
    Args:
        args: (config_id, params, n_episodes, eval_episodes)
    
    Returns:
        Results dictionary with params and metrics
    """
    config_id, params, n_episodes, eval_episodes = args
    
    # Set up environment
    env = mining_env.MiningEnv(
        board_shape=(10, 13),
        max_energy=95,
        minor_rewards=params["minor_rewards"]
    )
    
    # Create agent with these hyperparameters
    agent = SmartAgent(
        board_shape=(10, 13),
        n_retrieved=4,
        lr=params["lr"],
        gamma=params["gamma"],
        entropy_coef=params["entropy_coef"],
        conv_channels=params["conv_channels"],
        n_conv_layers=params["n_conv_layers"],
        fc_hidden_size=params["fc_hidden_size"],
        device=None  # Auto-detect (use GPU if available)
    )
    
    # Training
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        agent.prev_potential = None
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            shaped_reward = agent.shape_reward(reward, state, next_state, done)
            agent.store_transition(state, action, log_prob, shaped_reward, value, done)
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Update at end of episode
        with torch.no_grad():
            dust = torch.tensor(state["dust"]).unsqueeze(0).to(agent.device)
            energy = torch.tensor([state["energy"]]).to(agent.device)
            retrieved = torch.tensor(state["retrieved"]).unsqueeze(0).to(agent.device)
            next_value = agent.value(dust, energy, retrieved).item()
        agent.update(next_value)
    
    training_time = time.time() - start_time
    
    # Evaluation
    eval_metrics = evaluate_agent(agent, env, n_episodes=eval_episodes)
    
    # Compute training statistics
    avg_reward_first_half = np.mean(episode_rewards[:n_episodes//2])
    avg_reward_second_half = np.mean(episode_rewards[n_episodes//2:])
    improvement_during_training = avg_reward_second_half - avg_reward_first_half
    
    # Count model parameters
    n_params = sum(p.numel() for p in agent.policy.parameters()) + \
               sum(p.numel() for p in agent.value.parameters())
    
    result = {
        "config_id": config_id,
        "params": params,
        "eval_mean_reward": eval_metrics["mean_reward"],
        "eval_std_reward": eval_metrics["std_reward"],
        "eval_max_reward": eval_metrics["max_reward"],
        "eval_mean_retrieved": eval_metrics["mean_retrieved"],
        "training_improvement": improvement_during_training,
        "final_training_avg": avg_reward_second_half,
        "training_time": training_time,
        "n_params": n_params,
    }
    
    # Build compact param string
    arch_str = f"ch={params['conv_channels']} L={params['n_conv_layers']} fc={params['fc_hidden_size']}"
    print(f"Config {config_id:3d} | "
          f"lr={params['lr']:.0e} γ={params['gamma']:.2f} "
          f"ent={params['entropy_coef']:.2f} | {arch_str} | "
          f"Eval: {eval_metrics['mean_reward']:.2f} | "
          f"Params: {n_params/1000:.1f}K")
    
    return result


def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def get_csv_columns() -> List[str]:
    """Get CSV column names in consistent order."""
    return [
        "config_id",
        # Hyperparameters
        "lr", "gamma", "entropy_coef", "minor_rewards",
        "conv_channels", "n_conv_layers", "fc_hidden_size",
        # Results
        "eval_mean_reward", "eval_std_reward", "eval_max_reward",
        "eval_mean_retrieved", "training_improvement", "final_training_avg",
        "training_time", "n_params"
    ]


def result_to_csv_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten result dict for CSV writing."""
    row = {"config_id": result["config_id"]}
    # Add params
    for key, value in result["params"].items():
        row[key] = value
    # Add metrics
    for key in ["eval_mean_reward", "eval_std_reward", "eval_max_reward",
                "eval_mean_retrieved", "training_improvement", "final_training_avg",
                "training_time", "n_params"]:
        row[key] = result[key]
    return row


def write_csv_header(csv_path: str):
    """Write CSV header."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_columns())
        writer.writeheader()


def append_result_to_csv(csv_path: str, result: Dict[str, Any]):
    """Append a single result to CSV file."""
    row = result_to_csv_row(result)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_columns())
        writer.writerow(row)


def run_grid_search(
    n_workers: int,
    n_episodes: int,
    eval_episodes: int,
    param_grid: Dict[str, List],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Run parallel grid search with incremental CSV logging.
    
    Args:
        n_workers: Number of parallel workers
        n_episodes: Training episodes per config
        eval_episodes: Evaluation episodes per config
        param_grid: Dictionary of parameter lists
        output_dir: Directory to save results
    
    Returns:
        List of result dictionaries
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV file for incremental logging
    csv_path = os.path.join(output_dir, "results.csv")
    write_csv_header(csv_path)
    print(f"Results will be logged to: {csv_path}")
    
    # Generate all parameter combinations
    param_combos = generate_param_combinations(param_grid)
    n_configs = len(param_combos)
    
    print(f"Running grid search with {n_configs} configurations")
    print(f"Using {n_workers} parallel workers")
    print(f"Training episodes: {n_episodes}, Eval episodes: {eval_episodes}")
    print("=" * 80)
    
    # Prepare arguments for parallel execution
    args_list = [
        (i, params, n_episodes, eval_episodes)
        for i, params in enumerate(param_combos)
    ]
    
    # Run in parallel with incremental result collection
    start_time = time.time()
    results = []
    completed = 0
    
    if n_workers == 1:
        # Sequential execution for debugging
        for args in args_list:
            result = train_single_config(args)
            results.append(result)
            append_result_to_csv(csv_path, result)
            completed += 1
            print(f"  [{completed}/{n_configs}] Logged to CSV")
    else:
        # Parallel execution with imap_unordered for streaming results
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(train_single_config, args_list):
                results.append(result)
                append_result_to_csv(csv_path, result)
                completed += 1
                # Print progress less frequently to avoid clutter
                if completed % max(1, n_configs // 20) == 0 or completed == n_configs:
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (n_configs - completed) if completed > 0 else 0
                    print(f"  Progress: {completed}/{n_configs} ({100*completed/n_configs:.0f}%) | "
                          f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
    
    total_time = time.time() - start_time
    
    print("=" * 80)
    print(f"Grid search completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Sort by evaluation reward
    results.sort(key=lambda x: x["eval_mean_reward"], reverse=True)
    
    # Save results
    results_file = os.path.join(output_dir, "grid_search_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def print_top_results(results: List[Dict[str, Any]], top_k: int = 10):
    """Print top K configurations."""
    print(f"\n{'='*80}")
    print(f"TOP {min(top_k, len(results))} CONFIGURATIONS")
    print("=" * 80)
    
    for i, result in enumerate(results[:top_k]):
        params = result["params"]
        print(f"\n#{i+1} | Eval Reward: {result['eval_mean_reward']:.2f} ± {result['eval_std_reward']:.2f}")
        print(f"   | Retrieved: {result['eval_mean_retrieved']:.2f} | Max: {result['eval_max_reward']:.2f}")
        print(f"   | Learning: lr={params['lr']:.0e}, γ={params['gamma']}, entropy={params['entropy_coef']}")
        print(f"   | Network: conv_ch={params['conv_channels']}, layers={params['n_conv_layers']}, fc={params['fc_hidden_size']}")
        print(f"   | Model size: {result['n_params']/1000:.1f}K params | Training improvement: {result['training_improvement']:.2f}")


def analyze_parameter_importance(results: List[Dict[str, Any]], param_grid: Dict[str, List]):
    """Analyze which parameters have the most impact."""
    print(f"\n{'='*80}")
    print("PARAMETER IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Group parameters by category
    learning_params = ["lr", "gamma", "entropy_coef", "minor_rewards"]
    arch_params = ["conv_channels", "n_conv_layers", "fc_hidden_size"]
    
    print("\n--- Learning Parameters ---")
    for param_name in learning_params:
        if param_name not in param_grid:
            continue
        print(f"\n{param_name}:")
        for value in param_grid[param_name]:
            matching = [r for r in results if r["params"][param_name] == value]
            if matching:
                avg_reward = np.mean([r["eval_mean_reward"] for r in matching])
                std_reward = np.std([r["eval_mean_reward"] for r in matching])
                print(f"  {value}: avg reward = {avg_reward:.2f} ± {std_reward:.2f}")
    
    print("\n--- Network Architecture ---")
    for param_name in arch_params:
        if param_name not in param_grid:
            continue
        print(f"\n{param_name}:")
        for value in param_grid[param_name]:
            matching = [r for r in results if r["params"][param_name] == value]
            if matching:
                avg_reward = np.mean([r["eval_mean_reward"] for r in matching])
                std_reward = np.std([r["eval_mean_reward"] for r in matching])
                avg_params = np.mean([r["n_params"] for r in matching])
                print(f"  {value}: avg reward = {avg_reward:.2f} ± {std_reward:.2f} (avg {avg_params/1000:.1f}K params)")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Number of parallel workers (default: CPU count - 1)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Training episodes per config (default: 1000)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Evaluation episodes per config (default: 50)")
    parser.add_argument("--output-dir", type=str, default="hyperparam_search",
                        help="Output directory (default: hyperparam_search)")
    
    # Learning hyperparameters
    parser.add_argument("--lr", nargs="+", type=float, 
                        help="Learning rates to try")
    parser.add_argument("--gamma", nargs="+", type=float,
                        help="Gamma values to try")
    parser.add_argument("--entropy-coef", nargs="+", type=float,
                        help="Entropy coefficients to try")
    parser.add_argument("--minor-rewards", nargs="+", type=float,
                        help="Minor reward values to try")
    
    # Network architecture hyperparameters
    parser.add_argument("--conv-channels", nargs="+", type=int,
                        help="Conv layer channel sizes to try (e.g., 32 64 128)")
    parser.add_argument("--n-conv-layers", nargs="+", type=int,
                        help="Number of conv layers to try (e.g., 2 3 4)")
    parser.add_argument("--fc-hidden-size", nargs="+", type=int,
                        help="FC hidden layer sizes to try (e.g., 64 128 256)")
    
    args = parser.parse_args()
    
    # Set number of workers
    n_workers = args.workers or max(1, cpu_count() - 1)
    
    # Build parameter grid
    param_grid = PARAM_GRID.copy()
    if args.lr:
        param_grid["lr"] = args.lr
    if args.gamma:
        param_grid["gamma"] = args.gamma
    if args.entropy_coef:
        param_grid["entropy_coef"] = args.entropy_coef
    if args.minor_rewards:
        param_grid["minor_rewards"] = args.minor_rewards
    if args.conv_channels:
        param_grid["conv_channels"] = args.conv_channels
    if args.n_conv_layers:
        param_grid["n_conv_layers"] = args.n_conv_layers
    if args.fc_hidden_size:
        param_grid["fc_hidden_size"] = args.fc_hidden_size
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"search_{timestamp}")
    
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print(f"\nSearch space:")
    print("\n  Learning parameters:")
    for param in ["lr", "gamma", "entropy_coef", "minor_rewards"]:
        if param in param_grid:
            print(f"    {param}: {param_grid[param]}")
    print("\n  Network architecture:")
    for param in ["conv_channels", "n_conv_layers", "fc_hidden_size"]:
        if param in param_grid:
            print(f"    {param}: {param_grid[param]}")
    
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)
    print(f"\nTotal configurations: {n_combos}")
    est_time_sec = n_combos * args.episodes * 0.03 / n_workers
    print(f"Estimated time: ~{est_time_sec:.0f}s ({est_time_sec/60:.1f} min)")
    print()
    
    # Run grid search
    results = run_grid_search(
        n_workers=n_workers,
        n_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        param_grid=param_grid,
        output_dir=output_dir
    )
    
    # Print analysis
    print_top_results(results)
    analyze_parameter_importance(results, param_grid)
    
    # Print best configuration
    best = results[0]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Eval Reward: {best['eval_mean_reward']:.2f} ± {best['eval_std_reward']:.2f}")
    print(f"Mean Retrieved: {best['eval_mean_retrieved']:.2f}")
    print(f"Model Size: {best['n_params']/1000:.1f}K parameters")
    print(f"\nParameters:")
    for param, value in best["params"].items():
        print(f"  --{param.replace('_', '-')} {value}")
    
    print(f"\nResults saved to {output_dir}/")
    
    # Save best config as a runnable command
    cmd_file = os.path.join(output_dir, "best_config.sh")
    with open(cmd_file, "w") as f:
        params = best["params"]
        f.write("#!/bin/bash\n")
        f.write("# Best hyperparameter configuration\n")
        f.write(f"# Model size: {best['n_params']/1000:.1f}K parameters\n")
        f.write(f"# Eval reward: {best['eval_mean_reward']:.2f}\n\n")
        f.write(f"python train_smart.py \\\n")
        f.write(f"  --lr {params['lr']} \\\n")
        f.write(f"  --gamma {params['gamma']} \\\n")
        f.write(f"  --entropy-coef {params['entropy_coef']} \\\n")
        f.write(f"  --minor-rewards {params['minor_rewards']} \\\n")
        f.write(f"  --conv-channels {params['conv_channels']} \\\n")
        f.write(f"  --n-conv-layers {params['n_conv_layers']} \\\n")
        f.write(f"  --fc-hidden-size {params['fc_hidden_size']} \\\n")
        f.write(f"  --episodes 2000\n")
    
    print(f"Best config command saved to {cmd_file}")


if __name__ == "__main__":
    main()
