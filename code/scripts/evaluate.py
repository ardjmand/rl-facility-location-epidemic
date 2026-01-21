"""
Evaluation script for trained PPO policy on epidemic facility location.

Evaluates trained models on test problems and compares against baselines.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.pt [options]

Example:
    python scripts/evaluate.py --checkpoint checkpoints/final_1000000.pt --num-episodes 100
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd

from params import device as default_device
from funcs import EpidemicEnvironment, generate_problem_set
from rl import (
    PPOConfig,
    EpidemicGymEnv,
    GNNActorCritic,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO policy on epidemic facility location"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint"
    )

    # Evaluation settings
    parser.add_argument(
        "--num-episodes", type=int, default=100,
        help="Number of evaluation episodes (default: 100)"
    )
    parser.add_argument(
        "--problem-dir", type=str, default=None,
        help="Directory with saved test problems (generates new if not provided)"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions (default: True)"
    )
    parser.add_argument(
        "--no-deterministic", action="store_false", dest="deterministic",
        help="Use stochastic actions"
    )

    # Environment settings
    parser.add_argument(
        "--decision-interval", type=int, default=100,
        help="Epidemic timesteps between RL decisions (default: 100)"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=30,
        help="Maximum decisions per episode (default: 30)"
    )

    # Baselines
    parser.add_argument(
        "--run-baselines", action="store_true", default=True,
        help="Run baseline comparisons (default: True)"
    )
    parser.add_argument(
        "--no-baselines", action="store_false", dest="run_baselines",
        help="Skip baseline comparisons"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./eval_results",
        help="Directory for evaluation results (default: ./eval_results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)"
    )

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        return default_device
    return torch.device(device_str)


def load_policy(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[GNNActorCritic, Dict]:
    """
    Load trained policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device

    Returns:
        Tuple of (policy, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    ppo_config = checkpoint.get('config', {}).get('ppo', {})

    # Create policy
    policy = GNNActorCritic(
        individual_in_dim=5,
        facility_in_dim=3,
        hidden_dim=ppo_config.get('hidden_dim', 128),
        num_gnn_layers=ppo_config.get('num_gnn_layers', 3),
        global_feature_dim=ppo_config.get('global_feature_dim', 8),
        dropout=ppo_config.get('dropout', 0.1),
    ).to(device)

    # Load weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    return policy, checkpoint


def run_episode(
    env: EpidemicGymEnv,
    policy: Optional[GNNActorCritic],
    strategy: str = "policy",
    deterministic: bool = True,
) -> Dict:
    """
    Run a single evaluation episode.

    Args:
        env: Gymnasium environment
        policy: Trained policy (None for baselines)
        strategy: "policy", "random", "all_open", "all_closed", "fixed_initial"
        deterministic: Use deterministic actions

    Returns:
        Dictionary with episode results
    """
    obs, info = env.reset()
    done = False

    total_reward = 0.0
    total_cost = 0.0
    step_count = 0
    total_infections = 0
    total_vaccinations = 0
    facility_changes = 0

    M = obs['num_facilities']
    initial_status = obs['facility_features'][:M, 0].copy()

    while not done:
        # Choose action based on strategy
        if strategy == "policy":
            with torch.no_grad():
                action, _, _, _ = policy(obs, deterministic=deterministic)
            action = action.cpu().numpy()
        elif strategy == "random":
            action = np.random.randint(0, 2, size=M)
        elif strategy == "all_open":
            action = np.ones(M, dtype=np.int32)
        elif strategy == "all_closed":
            action = np.zeros(M, dtype=np.int32)
        elif strategy == "fixed_initial":
            action = initial_status.astype(np.int32)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_cost += step_info.get('step_cost', 0)
        step_count += 1

    # Get final info
    episode_info = step_info.get('episode_info', {})

    return {
        'strategy': strategy,
        'total_reward': total_reward,
        'total_cost': total_cost,
        'episode_length': step_count,
        'total_infections': episode_info.get('total_infections', 0),
        'total_vaccinations': episode_info.get('total_vaccinations', 0),
        'facility_changes': episode_info.get('facility_changes', 0),
        'N': episode_info.get('N', 0),
        'M': episode_info.get('M', 0),
        'terminated_early': step_count < env.max_episode_steps,
    }


def evaluate_strategy(
    env: EpidemicGymEnv,
    policy: Optional[GNNActorCritic],
    strategy: str,
    num_episodes: int,
    deterministic: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate a strategy over multiple episodes.

    Args:
        env: Gymnasium environment
        policy: Trained policy
        strategy: Strategy name
        num_episodes: Number of episodes
        deterministic: Use deterministic actions
        seed: Random seed

    Returns:
        DataFrame with episode results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    for i in tqdm(range(num_episodes), desc=f"Evaluating {strategy}"):
        result = run_episode(env, policy, strategy, deterministic)
        result['episode'] = i
        results.append(result)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, strategy: str):
    """Print summary statistics for a strategy."""
    print(f"\n{strategy.upper()} Results:")
    print("-" * 40)
    print(f"  Mean Reward:     {df['total_reward'].mean():>10.2f} +/- {df['total_reward'].std():.2f}")
    print(f"  Mean Cost:       {df['total_cost'].mean():>10.2f} +/- {df['total_cost'].std():.2f}")
    print(f"  Mean Length:     {df['episode_length'].mean():>10.2f}")
    print(f"  Mean Infections: {df['total_infections'].mean():>10.2f}")
    print(f"  Mean Vaccinations: {df['total_vaccinations'].mean():>8.2f}")
    print(f"  Mean Fac Changes: {df['facility_changes'].mean():>9.2f}")
    print(f"  Early Termination: {df['terminated_early'].mean() * 100:>7.1f}%")


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    print(f"\nLoading policy from: {args.checkpoint}")
    policy, checkpoint = load_policy(args.checkpoint, device)

    # Print checkpoint info
    print(f"  Trained for: {checkpoint.get('timesteps_collected', 'unknown')} timesteps")
    print(f"  Episodes: {checkpoint.get('episodes_completed', 'unknown')}")
    print(f"  Saved at: {checkpoint.get('saved_at', 'unknown')}")

    # Create environment
    print("\nCreating evaluation environment...")
    env = EpidemicGymEnv(
        randomize=True,  # Use randomization for diverse test cases
        decision_interval=args.decision_interval,
        max_episode_steps=args.max_episode_steps,
        terminate_on_resolution=True,
        device=device,
    )

    # Evaluate trained policy
    print(f"\nEvaluating trained policy on {args.num_episodes} episodes...")
    policy_results = evaluate_strategy(
        env=env,
        policy=policy,
        strategy="policy",
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    print_summary(policy_results, "Trained Policy")

    all_results = [policy_results]

    # Run baselines
    if args.run_baselines:
        baselines = ["random", "all_open", "all_closed", "fixed_initial"]

        for baseline in baselines:
            print(f"\nEvaluating {baseline} baseline...")
            baseline_results = evaluate_strategy(
                env=env,
                policy=None,
                strategy=baseline,
                num_episodes=args.num_episodes,
                deterministic=True,
                seed=args.seed,
            )
            print_summary(baseline_results, baseline)
            all_results.append(baseline_results)

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"eval_results_{timestamp}.csv"
    combined_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    comparison = combined_df.groupby('strategy').agg({
        'total_reward': ['mean', 'std'],
        'total_cost': ['mean', 'std'],
        'episode_length': 'mean',
        'total_infections': 'mean',
    }).round(2)

    print(comparison)

    # Save summary
    summary_path = output_dir / f"eval_summary_{timestamp}.json"
    summary = {
        'checkpoint': args.checkpoint,
        'num_episodes': args.num_episodes,
        'deterministic': args.deterministic,
        'timestamp': timestamp,
        'results': {}
    }

    for strategy in combined_df['strategy'].unique():
        strategy_df = combined_df[combined_df['strategy'] == strategy]
        summary['results'][strategy] = {
            'mean_reward': float(strategy_df['total_reward'].mean()),
            'std_reward': float(strategy_df['total_reward'].std()),
            'mean_cost': float(strategy_df['total_cost'].mean()),
            'std_cost': float(strategy_df['total_cost'].std()),
            'mean_length': float(strategy_df['episode_length'].mean()),
            'mean_infections': float(strategy_df['total_infections'].mean()),
            'mean_vaccinations': float(strategy_df['total_vaccinations'].mean()),
        }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
