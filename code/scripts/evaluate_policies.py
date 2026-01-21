"""
Flexible evaluation script for testing policies on epidemic control problems.

Supports:
- Multiple trained checkpoints
- Static baseline strategies (all_open, all_closed, random, fixed_initial)
- Random or saved problem sets
- Detailed summary output with rankings and pairwise comparisons

Usage:
    # Test PPO checkpoint against baselines on random problems
    python scripts/evaluate_policies.py --checkpoints checkpoints/final.pt --num-problems 100

    # Test on saved problems
    python scripts/evaluate_policies.py --checkpoints checkpoints/final.pt --problem-dir saved_problems/

    # Multiple checkpoints, specific baselines
    python scripts/evaluate_policies.py --checkpoints ckpt1.pt ckpt2.pt --baselines all_open all_closed
"""

import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import EpidemicGymEnv
from funcs import EpidemicEnvironment, generate_problem_set
from policies import (
    Policy,
    STATIC_POLICIES,
    load_checkpoint,
)


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""
    policy_name: str
    problem_idx: int
    total_reward: float
    total_cost: float
    episode_length: int
    total_infections: float
    total_vaccinations: float
    facility_opens: int
    facility_closes: int
    mean_open_fraction: float  # Average fraction of facilities open during episode
    terminated_early: bool
    N: int
    M: int


@dataclass
class EvaluationResults:
    """Aggregated results from evaluation."""
    episodes: List[EpisodeResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([vars(ep) for ep in self.episodes])

    def get_policy_stats(self) -> pd.DataFrame:
        """Get summary statistics per policy."""
        df = self.to_dataframe()
        stats = df.groupby("policy_name").agg({
            "total_cost": ["mean", "std"],
            "total_reward": ["mean", "std"],
            "total_infections": ["mean", "std"],
            "total_vaccinations": ["mean", "std"],
            "facility_opens": ["mean", "std"],
            "facility_closes": ["mean", "std"],
            "mean_open_fraction": ["mean", "std"],
            "episode_length": ["mean", "std"],
            "terminated_early": "mean",
        })
        # Flatten column names
        stats.columns = ["_".join(col).strip("_") for col in stats.columns]
        return stats.reset_index()


@dataclass
class ProblemInfo:
    """Information about a saved problem."""
    idx: int
    filepath: Path
    N: int
    M: int
    avg_deg: float = 0.0
    # Epidemic parameters
    beta_1: float = 0.0
    beta_2: float = 0.0
    delta: float = 0.0
    omega: float = 0.0
    # Vaccination parameters
    v_max: float = 0.0
    v_min: float = 0.0
    alpha: float = 0.0
    # Cost parameters
    C_I: float = 0.0
    C_V: float = 0.0
    C_O: float = 0.0
    f_plus: float = 0.0
    f_minus: float = 0.0


def generate_and_save_problems(
    num_problems: int,
    output_dir: Path,
    seed: int,
    device: torch.device,
    verbose: bool = True,
) -> List[ProblemInfo]:
    """Generate random problems and save them to disk.

    Args:
        num_problems: Number of problems to generate
        output_dir: Directory to save problems
        seed: Random seed for reproducibility
        device: Device for tensors
        verbose: Show progress

    Returns:
        List of ProblemInfo with file paths and metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating {num_problems} random problems...")

    # Generate problems using the existing utility
    filepaths = generate_problem_set(
        output_dir=str(output_dir),
        num_problems=num_problems,
        randomize=True,
        seed=seed,
        verbose=verbose,
    )

    # Collect problem info
    problems = []
    for idx, filepath in enumerate(filepaths):
        env = EpidemicEnvironment.load(filepath, device=device)
        problems.append(ProblemInfo(
            idx=idx,
            filepath=Path(filepath),
            N=env.N,
            M=env.M,
            avg_deg=float(env.avg_deg),
            beta_1=float(env.beta_1),
            beta_2=float(env.beta_2),
            delta=float(env.delta),
            omega=float(env.omega),
            v_max=float(env.v_max),
            v_min=float(env.v_min),
            alpha=float(env.alpha),
            C_I=float(env.C_I),
            C_V=float(env.C_V),
            C_O=float(env.C_O),
            f_plus=float(env.f_plus),
            f_minus=float(env.f_minus),
        ))

    return problems


def load_saved_problems(
    problem_dir: Path,
    pattern: str,
    device: torch.device,
    verbose: bool = True,
) -> List[ProblemInfo]:
    """Load problem info from saved files.

    Args:
        problem_dir: Directory containing saved problems
        pattern: Glob pattern for problem files
        device: Device for tensors
        verbose: Show progress

    Returns:
        List of ProblemInfo with file paths and metadata
    """
    filepaths = sorted(problem_dir.glob(pattern))

    if verbose:
        print(f"Found {len(filepaths)} saved problems in {problem_dir}")

    problems = []
    for idx, filepath in enumerate(tqdm(filepaths, desc="Loading problem info", disable=not verbose)):
        env = EpidemicEnvironment.load(filepath, device=device)
        problems.append(ProblemInfo(
            idx=idx,
            filepath=filepath,
            N=env.N,
            M=env.M,
            avg_deg=float(env.avg_deg),
            beta_1=float(env.beta_1),
            beta_2=float(env.beta_2),
            delta=float(env.delta),
            omega=float(env.omega),
            v_max=float(env.v_max),
            v_min=float(env.v_min),
            alpha=float(env.alpha),
            C_I=float(env.C_I),
            C_V=float(env.C_V),
            C_O=float(env.C_O),
            f_plus=float(env.f_plus),
            f_minus=float(env.f_minus),
        ))

    return problems


def run_episode(
    env: EpidemicGymEnv,
    policy: Policy,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Run a single evaluation episode.

    Args:
        env: Gymnasium environment (fresh instance loaded from saved problem)
        policy: Policy to evaluate
        deterministic: Use deterministic action selection

    Returns:
        Dictionary with episode metrics
    """
    # Don't call env.reset() - it would regenerate with default params.
    # Instead, manually initialize episode state and get observation.
    env._current_step = 0
    env._cumulative_cost = 0.0
    env._episode_info = {
        "total_infections": 0,
        "total_vaccinations": 0,
        "facility_changes": 0,
        "N": env.epidemic_env.N,
        "M": env.epidemic_env.M,
    }
    obs = env._get_observation()
    policy.reset(obs)

    total_reward = 0.0
    total_cost = 0.0
    facility_opens = 0
    facility_closes = 0
    total_open_fraction = 0.0
    steps = 0

    done = False
    prev_action = None
    last_step_info = None
    M = obs["num_facilities"]

    while not done:
        action = policy.select_action(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, step_info = env.step(action)
        last_step_info = step_info

        total_reward += reward
        total_cost += step_info.get("step_cost", -reward)

        # Track open fraction
        total_open_fraction += np.sum(action) / M

        # Count facility changes
        if prev_action is not None:
            facility_opens += int(np.sum((action == 1) & (prev_action == 0)))
            facility_closes += int(np.sum((action == 0) & (prev_action == 1)))
        prev_action = action.copy()

        steps += 1
        done = terminated or truncated

    # Get cumulative stats from episode_info
    episode_info = last_step_info.get("episode_info", {}) if last_step_info else {}
    total_infections = episode_info.get("total_infections", 0)
    total_vaccinations = episode_info.get("total_vaccinations", 0)
    mean_open_fraction = total_open_fraction / steps if steps > 0 else 0.0

    return {
        "total_reward": total_reward,
        "total_cost": total_cost,
        "episode_length": steps,
        "total_infections": total_infections,
        "total_vaccinations": total_vaccinations,
        "facility_opens": facility_opens,
        "facility_closes": facility_closes,
        "mean_open_fraction": mean_open_fraction,
        "terminated_early": terminated,
        "N": obs["num_individuals"],
        "M": obs["num_facilities"],
    }


def evaluate_policies(
    policies: List[Policy],
    problems: List[ProblemInfo],
    decision_interval: int,
    max_episode_steps: int,
    device: torch.device,
    deterministic: bool = True,
    runs_per_problem: int = 1,
    verbose: bool = True,
) -> EvaluationResults:
    """Evaluate multiple policies on a set of problems.

    For each (policy, problem) pair, loads a FRESH copy of the problem
    to ensure all policies see the exact same initial state.

    Args:
        policies: List of policies to evaluate
        problems: List of ProblemInfo with saved problem paths
        decision_interval: Timesteps between decisions
        max_episode_steps: Maximum steps per episode
        device: Device for tensors
        deterministic: Use deterministic action selection
        runs_per_problem: Number of runs per problem (for stochastic policies)
        verbose: Show progress bar

    Returns:
        EvaluationResults with all episode data
    """
    results = EvaluationResults()

    if verbose:
        print(f"\nEvaluating {len(policies)} policies on {len(problems)} problems")
        print("Problem sizes:")
        for p in problems:
            print(f"  Problem {p.idx}: N={p.N}, M={p.M}")
        print()

    # Total evaluations
    total_evals = len(policies) * len(problems) * runs_per_problem
    pbar = tqdm(total=total_evals, desc="Evaluating", disable=not verbose)

    for policy in policies:
        for problem in problems:
            for run in range(runs_per_problem):
                # Load a FRESH copy of the problem for this evaluation
                # This ensures the initial state is identical for all policies
                env = EpidemicGymEnv.from_saved_problem(
                    filepath=str(problem.filepath),
                    decision_interval=decision_interval,
                    max_episode_steps=max_episode_steps,
                    device=device,
                )

                # Run episode on fresh environment
                metrics = run_episode(env, policy, deterministic=deterministic)

                # Record result
                results.episodes.append(EpisodeResult(
                    policy_name=policy.name,
                    problem_idx=problem.idx,
                    **metrics
                ))

                pbar.update(1)

    pbar.close()
    return results


def compute_win_rates(results: EvaluationResults) -> Dict[str, float]:
    """Compute win rate for each policy (lowest cost wins).

    Args:
        results: Evaluation results

    Returns:
        Dictionary mapping policy name to win rate (0-1)
    """
    df = results.to_dataframe()

    # Group by problem and find winner (lowest cost)
    problem_winners = df.loc[df.groupby("problem_idx")["total_cost"].idxmin()]
    winner_counts = problem_winners["policy_name"].value_counts()

    num_problems = df["problem_idx"].nunique()
    win_rates = {}
    for policy in df["policy_name"].unique():
        wins = winner_counts.get(policy, 0)
        win_rates[policy] = wins / num_problems

    return win_rates


def compute_pairwise_comparison(results: EvaluationResults) -> pd.DataFrame:
    """Compute pairwise cost differences between policies.

    Args:
        results: Evaluation results

    Returns:
        DataFrame with pairwise differences (row - column)
    """
    df = results.to_dataframe()

    # Pivot to get costs per policy per problem
    pivot = df.pivot_table(
        index="problem_idx",
        columns="policy_name",
        values="total_cost",
        aggfunc="mean"
    )

    policies = pivot.columns.tolist()
    comparison = pd.DataFrame(index=policies, columns=policies, dtype=float)

    for p1 in policies:
        for p2 in policies:
            if p1 == p2:
                comparison.loc[p1, p2] = 0.0
            else:
                # Positive means p1 costs more than p2
                comparison.loc[p1, p2] = (pivot[p1] - pivot[p2]).mean()

    return comparison


def print_summary(
    results: EvaluationResults,
    problems: List[ProblemInfo],
    num_problems: int,
    seed: int,
    decision_interval: int,
    problem_source_desc: str,
) -> None:
    """Print formatted evaluation summary.

    Args:
        results: Evaluation results
        problems: List of problem info with parameters
        num_problems: Number of problems evaluated
        seed: Random seed used
        decision_interval: Decision interval setting
        problem_source_desc: Description of problem source
    """
    stats = results.get_policy_stats()
    win_rates = compute_win_rates(results)

    # Header
    print("\n" + "=" * 120)
    print(" " * 45 + "EVALUATION SUMMARY")
    print("=" * 120)
    print(f"Problems: {num_problems} {problem_source_desc} | Seed: {seed} | Decision interval: {decision_interval}")
    print("=" * 120)

    # Problem parameters table
    print("\nPROBLEM PARAMETERS:")
    print("-" * 145)
    print(f"{'#':<4}{'N':>6}{'M':>4}{'deg':>6}{'beta_1':>8}{'beta_2':>8}{'delta':>7}{'omega':>7}{'v_max':>7}{'v_min':>8}{'alpha':>7}{'C_I':>8}{'C_V':>8}{'C_O':>8}{'f+':>7}{'f-':>7}")
    print("-" * 145)
    for p in problems:
        print(f"{p.idx:<4}{p.N:>6}{p.M:>4}{p.avg_deg:>6.1f}{p.beta_1:>8.3f}{p.beta_2:>8.3f}{p.delta:>7.3f}{p.omega:>7.3f}{p.v_max:>7.2f}{p.v_min:>8.5f}{p.alpha:>7.2f}{p.C_I:>8.2f}{p.C_V:>8.4f}{p.C_O:>8.4f}{p.f_plus:>7.2f}{p.f_minus:>7.2f}")
    print("-" * 145)

    # Sort by mean cost
    stats_sorted = stats.sort_values("total_cost_mean")

    # Single table with all metrics
    print("\nPOLICY RESULTS:")
    print("-" * 120)
    print(f"{'Policy':<22}{'Cost':>14}{'Infections':>14}{'Vaccinations':>14}{'Open %':>10}{'Win Rate':>10}{'Length':>10}")
    print("-" * 120)

    for _, row in stats_sorted.iterrows():
        policy = row["policy_name"]
        cost = f"{row['total_cost_mean']:.1f}"
        infections = f"{row['total_infections_mean']:.0f}"
        vaccinations = f"{row['total_vaccinations_mean']:.0f}"
        open_frac = f"{row['mean_open_fraction_mean'] * 100:.1f}%"
        win_rate = f"{win_rates.get(policy, 0) * 100:.1f}%"
        length = f"{row['episode_length_mean']:.1f}"

        print(f"{policy:<22}{cost:>14}{infections:>14}{vaccinations:>14}{open_frac:>10}{win_rate:>10}{length:>10}")

    print("=" * 120 + "\n")


def save_results(
    results: EvaluationResults,
    output_dir: Path,
    save_csv: bool = True,
    save_json: bool = True,
    timestamp: Optional[str] = None,
) -> Dict[str, Path]:
    """Save results to files.

    Args:
        results: Evaluation results
        output_dir: Directory to save to
        save_csv: Save CSV with all episode data
        save_json: Save JSON with summary statistics
        timestamp: Optional timestamp for filenames

    Returns:
        Dictionary mapping file type to saved path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = {}

    if save_csv:
        csv_path = output_dir / f"eval_results_{timestamp}.csv"
        results.to_dataframe().to_csv(csv_path, index=False)
        saved_files["csv"] = csv_path
        print(f"Saved CSV: {csv_path}")

    if save_json:
        json_path = output_dir / f"eval_summary_{timestamp}.json"
        stats = results.get_policy_stats()
        win_rates = compute_win_rates(results)

        summary = {
            "timestamp": timestamp,
            "num_episodes": len(results.episodes),
            "policies": {},
        }

        for _, row in stats.iterrows():
            policy = row["policy_name"]
            summary["policies"][policy] = {
                "mean_cost": float(row["total_cost_mean"]),
                "std_cost": float(row["total_cost_std"]),
                "mean_reward": float(row["total_reward_mean"]),
                "mean_infections": float(row["total_infections_mean"]),
                "mean_vaccinations": float(row["total_vaccinations_mean"]),
                "mean_open_fraction": float(row["mean_open_fraction_mean"]),
                "win_rate": float(win_rates.get(policy, 0)),
            }

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        saved_files["json"] = json_path
        print(f"Saved JSON: {json_path}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policies on epidemic control problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test checkpoint against all baselines
  python scripts/evaluate_policies.py --checkpoints checkpoints/final.pt --num-problems 100

  # Test on saved problems
  python scripts/evaluate_policies.py --checkpoints checkpoints/final.pt --problem-dir saved_problems/

  # Multiple checkpoints, specific baselines
  python scripts/evaluate_policies.py --checkpoints ckpt1.pt ckpt2.pt --baselines all_open random

  # Save results
  python scripts/evaluate_policies.py --checkpoints checkpoints/final.pt --save-csv --save-json
        """
    )

    # Problem source
    problem_group = parser.add_mutually_exclusive_group()
    problem_group.add_argument(
        "--num-problems", type=int, default=100,
        help="Number of random problems to generate (default: 100)"
    )
    problem_group.add_argument(
        "--problem-dir", type=Path, default=None,
        help="Directory containing saved problems (overrides --num-problems)"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.pt",
        help="Glob pattern for saved problems (default: *.pt)"
    )
    parser.add_argument(
        "--keep-problems", action="store_true",
        help="Keep generated problems after evaluation (saves to ./eval_problems/)"
    )

    # Policy selection
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", default=[],
        help="Checkpoint files to evaluate"
    )
    parser.add_argument(
        "--policy-type", type=str, default="ppo",
        choices=["ppo"],
        help="Type of learned policy (default: ppo)"
    )
    parser.add_argument(
        "--baselines", type=str, nargs="+", default=None,
        choices=list(STATIC_POLICIES.keys()),
        help="Static baselines to include (default: all)"
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Skip all static baselines"
    )

    # Evaluation settings
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic action selection (default: True)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic action selection"
    )
    parser.add_argument(
        "--runs-per-problem", type=int, default=1,
        help="Number of runs per problem for stochastic evaluation (default: 1)"
    )
    parser.add_argument(
        "--decision-interval", type=int, default=100,
        help="Timesteps between decisions (default: 100)"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=30,
        help="Maximum steps per episode (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory to save results (default: results/)"
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Save detailed results to CSV"
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save summary to JSON"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress bars"
    )

    args = parser.parse_args()

    # Handle deterministic/stochastic
    deterministic = not args.stochastic

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build policy list
    policies: List[Policy] = []

    # Load checkpoints
    for ckpt_path in args.checkpoints:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            continue
        try:
            policy = load_checkpoint(ckpt_path, device, args.policy_type)
            policies.append(policy)
            print(f"Loaded checkpoint: {ckpt_path.name}")
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    # Add baselines
    if not args.no_baselines:
        baseline_names = args.baselines if args.baselines else list(STATIC_POLICIES.keys())
        for name in baseline_names:
            policies.append(STATIC_POLICIES[name]())
            print(f"Added baseline: {name}")

    if not policies:
        print("Error: No policies to evaluate!")
        sys.exit(1)

    print(f"\nTotal policies to evaluate: {len(policies)}")

    # Set up problems
    temp_dir = None
    try:
        if args.problem_dir:
            # Use existing saved problems
            if not args.problem_dir.exists():
                print(f"Error: Problem directory not found: {args.problem_dir}")
                sys.exit(1)

            problems = load_saved_problems(
                problem_dir=args.problem_dir,
                pattern=args.pattern,
                device=device,
                verbose=not args.quiet,
            )
            problem_source_desc = f"from {args.problem_dir}"
        else:
            # Generate random problems and save to temp directory
            if args.keep_problems:
                problem_dir = Path("eval_problems")
            else:
                temp_dir = tempfile.mkdtemp(prefix="eval_problems_")
                problem_dir = Path(temp_dir)

            problems = generate_and_save_problems(
                num_problems=args.num_problems,
                output_dir=problem_dir,
                seed=args.seed,
                device=device,
                verbose=not args.quiet,
            )
            problem_source_desc = "random"

            if args.keep_problems:
                print(f"Problems saved to: {problem_dir}")

        if not problems:
            print("Error: No problems to evaluate!")
            sys.exit(1)

        # Run evaluation
        results = evaluate_policies(
            policies=policies,
            problems=problems,
            decision_interval=args.decision_interval,
            max_episode_steps=args.max_episode_steps,
            device=device,
            deterministic=deterministic,
            runs_per_problem=args.runs_per_problem,
            verbose=not args.quiet,
        )

        # Print summary
        print_summary(
            results=results,
            problems=problems,
            num_problems=len(problems),
            seed=args.seed,
            decision_interval=args.decision_interval,
            problem_source_desc=problem_source_desc,
        )

        # Save results if requested
        if args.save_csv or args.save_json:
            save_results(
                results=results,
                output_dir=args.output_dir,
                save_csv=args.save_csv,
                save_json=args.save_json,
            )

    finally:
        # Clean up temp directory
        if temp_dir and not args.keep_problems:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
