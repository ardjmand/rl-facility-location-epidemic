"""
RL module for epidemic facility location using PPO with GNN-based policy.

This module provides:
- EpidemicGymEnv: Gymnasium wrapper for the epidemic simulation
- GNNActorCritic: GNN-based actor-critic network
- PPOTrainer: Custom PPO training loop
- Configuration classes for hyperparameters
"""

from .config import PPOConfig, TrainingConfig, EvalConfig
from .gym_env import EpidemicGymEnv
from .gnn_networks import (
    HeteroGNNEncoder,
    ActorHead,
    CriticHead,
    GNNActorCritic,
)
from .rollout_buffer import RolloutBuffer, EpisodeBuffer, StepMetricsBuffer
from .ppo_trainer import PPOTrainer
from .vec_env import VecEnv

__all__ = [
    # Config
    'PPOConfig',
    'TrainingConfig',
    'EvalConfig',
    # Environment
    'EpidemicGymEnv',
    'VecEnv',
    # Networks
    'HeteroGNNEncoder',
    'ActorHead',
    'CriticHead',
    'GNNActorCritic',
    # Buffer
    'RolloutBuffer',
    'EpisodeBuffer',
    'StepMetricsBuffer',
    # Trainer
    'PPOTrainer',
]
