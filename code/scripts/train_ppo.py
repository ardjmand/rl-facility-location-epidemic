"""
Training script for PPO with GNN-based policy on epidemic facility location.

Usage:
    python scripts/train_ppo.py [options]

Example:
    python scripts/train_ppo.py --total-timesteps 100000 --seed 42
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random

from params import device as default_device
from rl import (
    PPOConfig,
    TrainingConfig,
    EpidemicGymEnv,
    VecEnv,
    GNNActorCritic,
    PPOTrainer,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO with GNN policy on epidemic facility location"
    )

    # Training settings
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total timesteps to train (default: 1000000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    # Environment settings
    parser.add_argument(
        "--randomize", action="store_true", default=True,
        help="Use domain randomization (default: True)"
    )
    parser.add_argument(
        "--no-randomize", action="store_false", dest="randomize",
        help="Disable domain randomization"
    )
    parser.add_argument(
        "--decision-interval", type=int, default=100,
        help="Epidemic timesteps between RL decisions (default: 100)"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=30,
        help="Maximum decisions per episode (default: 30)"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of parallel environments (default: 1, use >1 for vectorized)"
    )
    parser.add_argument(
        "--sequential", action="store_true", default=False,
        help="Run environments sequentially (prevents GPU overheating)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max environments to step concurrently (default: num-envs). "
             "Reduce to prevent GPU overheating while keeping multiple envs."
    )

    # PPO hyperparameters
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=128,
        help="Steps per rollout (default: 128)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default: 64)"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO epochs per rollout (default: 10)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE lambda (default: 0.95)"
    )
    parser.add_argument(
        "--clip-range", type=float, default=0.2,
        help="PPO clip range (default: 0.2)"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5,
        help="Value function coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy coefficient (default: 0.01)"
    )

    # GNN architecture
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="GNN hidden dimension (default: 128)"
    )
    parser.add_argument(
        "--num-gnn-layers", type=int, default=3,
        help="Number of GNN layers (default: 3)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (default: 0.1)"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--eval-freq", type=int, default=10000,
        help="Evaluate every N timesteps (default: 10000)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50000,
        help="Checkpoint every N timesteps (default: 50000)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory for logs (default: ./logs)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory for checkpoints (default: ./checkpoints)"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)"
    )

    # Resume training
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--tb-log-dir", type=str, default=None,
        help="TensorBoard log directory (for resuming with same plots)"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="JSONL log file path (for resuming with same log file)"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        return default_device
    return torch.device(device_str)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create configs
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        randomize=args.randomize,
        decision_interval=args.decision_interval,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create environment(s)
    if args.num_envs > 1:
        mode_str = "sequential" if args.sequential else f"max_concurrent={args.max_concurrent or args.num_envs}"
        print(f"Creating vectorized environment with {args.num_envs} envs ({mode_str})...")
        env = VecEnv(
            num_envs=args.num_envs,
            randomize=args.randomize,
            decision_interval=args.decision_interval,
            max_episode_steps=args.max_episode_steps,
            sequential=args.sequential,
            max_concurrent=args.max_concurrent,
        )
    else:
        print("Creating environment...")
        env = EpidemicGymEnv(
            randomize=args.randomize,
            decision_interval=args.decision_interval,
            max_episode_steps=args.max_episode_steps,
            terminate_on_resolution=True,
            device=device,
        )

    # Create policy
    print("Creating GNN actor-critic policy...")
    policy = GNNActorCritic(
        individual_in_dim=5,
        facility_in_dim=3,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        global_feature_dim=8,
        dropout=args.dropout,
    )

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    # Create or load trainer
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer = PPOTrainer.load_checkpoint(
            path=args.resume,
            env=env,
            policy=policy,
            device=device,
            tb_log_dir=args.tb_log_dir,
            log_file=args.log_file,
        )
    else:
        trainer = PPOTrainer(
            env=env,
            policy=policy,
            config=ppo_config,
            training_config=training_config,
            device=device,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            tb_log_dir=args.tb_log_dir,
            log_file=args.log_file,
        )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Num envs: {args.num_envs}")
    if args.num_envs > 1:
        if args.sequential:
            print(f"  Parallelism: sequential (prevents GPU overheating)")
        else:
            max_conc = args.max_concurrent or args.num_envs
            print(f"  Parallelism: max_concurrent={max_conc}")
    print(f"  Domain randomization: {args.randomize}")
    print(f"  Decision interval: {args.decision_interval} dt steps")
    print(f"  Max episode steps: {args.max_episode_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PPO epochs: {args.n_epochs}")
    print(f"  GNN layers: {args.num_gnn_layers}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print("=" * 60 + "\n")

    try:
        results = trainer.train(
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total timesteps: {results['total_timesteps']:,}")
        print(f"  Total episodes: {results['total_episodes']:,}")
        print(f"  Total time: {results['total_time']:.1f} seconds")
        print(f"  Final checkpoint: {results['final_checkpoint']}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save checkpoint on interrupt
        checkpoint_path = trainer.save_checkpoint(prefix="interrupted")
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
