"""
Configuration dataclasses for PPO training with GNN-based policy.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""

    # Core PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 128               # Steps per rollout (reduced for faster iterations)
    batch_size: int = 64             # Mini-batch size
    n_epochs: int = 10               # PPO epochs per rollout
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda
    clip_range: float = 0.2          # PPO clip ratio
    clip_range_vf: Optional[float] = None  # Value function clip (None = no clip)

    # Loss coefficients
    vf_coef: float = 0.5             # Value function loss coefficient
    ent_coef: float = 0.01           # Entropy bonus coefficient
    max_grad_norm: float = 0.5       # Gradient clipping norm

    # Network architecture
    hidden_dim: int = 128            # GNN hidden dimension
    num_gnn_layers: int = 3          # Number of GNN layers
    global_feature_dim: int = 8      # Global feature dimension
    dropout: float = 0.1             # Dropout rate

    # Training settings
    normalize_advantage: bool = True
    target_kl: Optional[float] = None  # Early stopping KL threshold

    # Learning rate schedule
    lr_schedule: str = "constant"    # "constant" or "linear"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    total_timesteps: int = 1_000_000
    eval_freq: int = 10000           # Evaluate every N timesteps
    eval_episodes: int = 10          # Episodes per evaluation
    checkpoint_freq: int = 50000     # Checkpoint every N timesteps
    log_interval: int = 1            # Log every N rollouts

    # Environment settings
    randomize: bool = True           # Domain randomization
    decision_interval: int = 100     # dt steps between decisions
    max_episode_steps: int = 30      # Max decisions per episode
    terminate_on_resolution: bool = True  # End episode when I=0

    # Reproducibility
    seed: Optional[int] = 42

    # Directories
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    # Device
    device: str = "auto"             # "auto", "cuda", "cpu"


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    num_episodes: int = 100          # Number of episodes to evaluate
    problem_dir: Optional[str] = None  # Directory with saved problems
    deterministic: bool = True       # Use deterministic actions
    save_results: bool = True        # Save evaluation results
    results_dir: str = "./eval_results"

    # Baselines to compare against
    baselines: List[str] = field(default_factory=lambda: [
        "random", "all_open", "all_closed", "fixed_initial"
    ])
