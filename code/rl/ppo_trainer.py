"""
Custom PPO trainer for GNN-based policy on epidemic environment.

Implements Proximal Policy Optimization with:
- Clipped surrogate objective
- Value function loss
- Entropy bonus
- GAE advantage estimation
- Periodic checkpointing and logging
- TensorBoard integration for live training curves
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .config import PPOConfig, TrainingConfig
from .gym_env import EpidemicGymEnv
from .vec_env import VecEnv
from .gnn_networks import GNNActorCritic
from .rollout_buffer import RolloutBuffer, EpisodeBuffer, StepMetricsBuffer


class PPOTrainer:
    """
    Custom PPO trainer for GNN-based policy on epidemic environment.

    Features:
    - Handles variable-size graphs
    - Custom rollout collection with episode tracking
    - GAE advantage estimation
    - Clipped surrogate objective
    - Periodic checkpointing
    - Comprehensive logging
    """

    def __init__(
        self,
        env: Union[EpidemicGymEnv, VecEnv],
        policy: GNNActorCritic,
        config: PPOConfig,
        training_config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        log_dir: str = "../outputs/logs",
        checkpoint_dir: str = "../outputs/checkpoints",
        tb_log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize the PPO trainer.

        Args:
            env: Gymnasium environment (single or vectorized)
            policy: GNN actor-critic network
            config: PPO hyperparameters
            training_config: Training configuration
            device: Torch device
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
            tb_log_dir: TensorBoard log directory (auto-generated if None)
            log_file: JSONL log file path (auto-generated if None)
        """
        self.env = env
        self.policy = policy
        self.config = config
        self.training_config = training_config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if using vectorized environment
        self.is_vec_env = isinstance(env, VecEnv)
        self.num_envs = env.num_envs if self.is_vec_env else 1

        # Move policy to device
        self.policy = self.policy.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

        # Learning rate scheduler
        if config.lr_schedule == "linear":
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1 - step / self.training_config.total_timesteps
            )
        else:
            self.lr_scheduler = None

        # Buffers
        self.rollout_buffer = RolloutBuffer(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        self.episode_buffer = EpisodeBuffer(max_episodes=100)
        self.step_metrics_buffer = StepMetricsBuffer(max_steps=10000)

        # Logging
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        self.tb_log_dir = tb_log_dir
        if TENSORBOARD_AVAILABLE:
            if tb_log_dir is None:
                self.tb_log_dir = str(self.log_dir / f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)

        # JSONL log file (auto-generated if not provided)
        if log_file is None:
            self.log_file = str(self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        else:
            self.log_file = log_file

        # Training state
        self.timesteps_collected = 0
        self.episodes_completed = 0
        self.updates_performed = 0
        self.start_time = None

        # Metrics history
        self.metrics_history: List[Dict[str, float]] = []

        # Current episode tracking (per-environment for vec_env)
        self._current_episode_rewards = [0.0] * self.num_envs
        self._current_episode_lengths = [0] * self.num_envs

    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_freq: Optional[int] = None,
        eval_episodes: int = 10,
        checkpoint_freq: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            total_timesteps: Total timesteps to train (overrides config)
            eval_freq: Evaluate every N timesteps
            eval_episodes: Episodes per evaluation
            checkpoint_freq: Checkpoint every N timesteps
            verbose: Print progress

        Returns:
            Training statistics
        """
        total_timesteps = total_timesteps or self.training_config.total_timesteps
        eval_freq = eval_freq or self.training_config.eval_freq
        checkpoint_freq = checkpoint_freq or self.training_config.checkpoint_freq

        self.start_time = time.time()

        # Initialize environment(s)
        if self.is_vec_env:
            observations = self.env.reset()  # List of observations
        else:
            obs, info = self.env.reset()
            observations = [obs]  # Wrap in list for uniform handling

        self._current_episode_rewards = [0.0] * self.num_envs
        self._current_episode_lengths = [0] * self.num_envs

        if verbose:
            print(f"Starting PPO training for {total_timesteps} timesteps")
            print(f"  Device: {self.device}")
            print(f"  Num envs: {self.num_envs}")
            print(f"  Log file: {self.log_file}")
            print(f"  Checkpoint dir: {self.checkpoint_dir}")
            if self.writer is not None:
                print(f"  TensorBoard: tensorboard --logdir {self.log_dir}")
            print("-" * 60)

        rollout_count = 0

        while self.timesteps_collected < total_timesteps:
            # Collect rollouts (returns updated observations for next iteration)
            steps_collected, observations = self._collect_rollouts(observations)
            rollout_count += 1

            # Get bootstrap values for all environments
            with torch.no_grad():
                if self.is_vec_env:
                    # Batched value estimation
                    last_values = self.policy.get_value_batched(observations)  # [num_envs]
                else:
                    last_values = self.policy.get_value(observations[0]).unsqueeze(0)

            # Compute advantages (uses last values for all envs)
            last_dones = [False] * self.num_envs  # We continue with current obs
            self.rollout_buffer.compute_returns_and_advantages_vec(
                last_values.cpu().numpy(), last_dones
            )

            # PPO update
            update_metrics = self._ppo_update()
            self.updates_performed += 1

            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Get episode stats
            episode_stats = self.episode_buffer.get_stats()

            # Get step metrics (cost components, compartments)
            step_stats = self.step_metrics_buffer.get_stats()

            # Combine metrics
            metrics = {
                'timesteps': self.timesteps_collected,
                'episodes': self.episodes_completed,
                'updates': self.updates_performed,
                'rollout': rollout_count,
                'time_elapsed': time.time() - self.start_time,
                **update_metrics,
                **episode_stats,
                **step_stats,
            }
            self.metrics_history.append(metrics)

            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

            # Log to TensorBoard
            self._log_to_tensorboard(metrics)

            # Print progress
            if verbose and rollout_count % self.training_config.log_interval == 0:
                self._print_progress(metrics)

            # Evaluation (uses single env from vec_env or the single env)
            if eval_freq > 0 and self.timesteps_collected % eval_freq < self.config.n_steps * self.num_envs:
                eval_metrics = self.evaluate(n_episodes=eval_episodes)
                if verbose:
                    print(f"  Eval: reward={eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
                # Reset environment(s) after eval
                if self.is_vec_env:
                    observations = self.env.reset()
                else:
                    obs, info = self.env.reset()
                    observations = [obs]
                self._current_episode_rewards = [0.0] * self.num_envs
                self._current_episode_lengths = [0] * self.num_envs

            # Checkpointing
            if checkpoint_freq > 0 and self.timesteps_collected % checkpoint_freq < self.config.n_steps * self.num_envs:
                checkpoint_path = self.save_checkpoint()
                if verbose:
                    print(f"  Saved checkpoint: {checkpoint_path}")

            # Clear buffer for next rollout
            self.rollout_buffer.clear()

        # Final checkpoint
        final_checkpoint = self.save_checkpoint(prefix="final")

        # Close TensorBoard writer
        self.close()

        if verbose:
            print("-" * 60)
            print(f"Training complete! Final checkpoint: {final_checkpoint}")

        return {
            'total_timesteps': self.timesteps_collected,
            'total_episodes': self.episodes_completed,
            'total_updates': self.updates_performed,
            'total_time': time.time() - self.start_time,
            'final_checkpoint': str(final_checkpoint),
        }

    def _collect_rollouts(
        self,
        observations: List[Dict[str, Any]]
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Collect experience by running policy in environment(s).

        Supports both single and vectorized environments.
        For vectorized environments, uses batched action selection and
        parallel environment stepping.

        Args:
            observations: List of current observations (one per env)

        Returns:
            Tuple of (number of steps collected, final observations)
        """
        steps = 0

        for _ in range(self.config.n_steps):
            with torch.no_grad():
                if self.is_vec_env:
                    # Batched action selection for all environments
                    actions, log_probs, _, values = self.policy.forward_batched(observations)
                    # actions is List[Tensor], log_probs and values are [num_envs] tensors
                    log_prob_vals = log_probs.cpu().numpy()  # [num_envs]
                    value_vals = values.cpu().numpy()  # [num_envs]
                else:
                    # Single environment
                    action, log_prob, _, value = self.policy(observations[0])
                    actions = [action]
                    log_prob_vals = [log_prob.item()]
                    value_vals = [value.item()]

            # Step environment(s)
            if self.is_vec_env:
                # Parallel step with auto-reset
                next_observations, rewards, terminateds, truncateds, infos, reset_infos = \
                    self.env.step_and_reset(actions)
                dones = [t or tr for t, tr in zip(terminateds, truncateds)]
            else:
                next_obs, reward, terminated, truncated, info = self.env.step(actions[0])
                next_observations = [next_obs]
                rewards = [reward]
                dones = [terminated or truncated]
                infos = [info]
                reset_infos = [{}]

            # Store transitions for each environment
            for env_idx in range(self.num_envs):
                self.rollout_buffer.add(
                    obs=observations[env_idx],
                    action=actions[env_idx],
                    log_prob=log_prob_vals[env_idx],
                    reward=rewards[env_idx],
                    value=value_vals[env_idx],
                    done=dones[env_idx],
                    env_idx=env_idx,  # Track which env this came from
                )

                # Track step metrics for TensorBoard
                N = observations[env_idx].get('num_individuals', 1)
                M = observations[env_idx].get('num_facilities', 1)
                self.step_metrics_buffer.add_step(infos[env_idx], N, M)

                # Update episode tracking for this env
                self._current_episode_rewards[env_idx] += rewards[env_idx]
                self._current_episode_lengths[env_idx] += 1

                if dones[env_idx]:
                    # Record episode
                    self.episode_buffer.add_episode(
                        total_reward=self._current_episode_rewards[env_idx],
                        length=self._current_episode_lengths[env_idx],
                        info=infos[env_idx].get('episode_info', {}),
                    )
                    self.episodes_completed += 1

                    # Reset tracking for this env (env already reset via step_and_reset)
                    self._current_episode_rewards[env_idx] = 0.0
                    self._current_episode_lengths[env_idx] = 0

            steps += self.num_envs
            self.timesteps_collected += self.num_envs
            observations = next_observations

        return steps, observations

    def _ppo_update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.

        Uses batched graph processing for efficiency when possible.
        Falls back to sequential processing for memory efficiency with large graphs.

        Returns:
            Dictionary with loss metrics
        """
        # Metrics accumulators
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []

        # Sub-batch size for graph batching (to avoid OOM with large graphs)
        # Process graphs in smaller chunks to fit in GPU memory
        graph_batch_size = min(8, self.config.batch_size)  # Max 8 graphs at a time

        # Multiple epochs
        for epoch in range(self.config.n_epochs):
            # Iterate over mini-batches
            for batch in self.rollout_buffer.get_batches(self.config.batch_size):
                batch_size = len(batch['indices'])
                observations = batch['observations']
                actions = batch['actions']

                # Ensure actions are tensors
                actions_tensors = []
                for action in actions:
                    if not isinstance(action, torch.Tensor):
                        actions_tensors.append(torch.tensor(action, dtype=torch.float32, device=self.device))
                    else:
                        actions_tensors.append(action.to(device=self.device, dtype=torch.float32))

                # Convert numpy arrays to tensors for batch processing
                old_log_probs = torch.tensor(batch['old_log_probs'], dtype=torch.float32, device=self.device)
                advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
                returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
                old_values = torch.tensor(batch['values'], dtype=torch.float32, device=self.device)

                # Normalize advantages (over full batch)
                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Process in sub-batches to avoid OOM
                all_new_log_probs = []
                all_entropies = []
                all_values = []

                for start_idx in range(0, batch_size, graph_batch_size):
                    end_idx = min(start_idx + graph_batch_size, batch_size)
                    sub_obs = observations[start_idx:end_idx]
                    sub_actions = actions_tensors[start_idx:end_idx]

                    # Batched evaluation for sub-batch
                    with torch.set_grad_enabled(True):
                        sub_log_probs, sub_entropies, sub_values = self.policy.evaluate_actions_batched(
                            sub_obs, sub_actions
                        )
                    all_new_log_probs.append(sub_log_probs)
                    all_entropies.append(sub_entropies)
                    all_values.append(sub_values)

                # Concatenate results
                new_log_probs = torch.cat(all_new_log_probs, dim=0)
                entropies = torch.cat(all_entropies, dim=0)
                values = torch.cat(all_values, dim=0)

                # Compute ratio
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.config.clip_range_vf is not None:
                    value_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss_unclipped = (values - returns) ** 2
                    value_loss_clipped = (value_clipped - returns) ** 2
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = ((values - returns) ** 2).mean()

                # Entropy loss (we want to maximize entropy, so negate)
                entropy_loss = -entropies.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.vf_coef * value_loss +
                    self.config.ent_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Record metrics (only call .item() here, after backward)
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((torch.abs(ratio - 1) > self.config.clip_range).float().mean()).item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropies.mean().item())
                total_losses.append(total_loss.item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_frac)

            # Early stopping on KL divergence
            if self.config.target_kl is not None:
                if np.mean(approx_kls) > self.config.target_kl:
                    break

        return {
            'policy_loss': float(np.mean(policy_losses)),
            'value_loss': float(np.mean(value_losses)),
            'entropy': float(np.mean(entropy_losses)),
            'total_loss': float(np.mean(total_losses)),
            'approx_kl': float(np.mean(approx_kls)),
            'clip_fraction': float(np.mean(clip_fractions)),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    def evaluate(
        self,
        eval_env: Optional[EpidemicGymEnv] = None,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate current policy.

        Args:
            eval_env: Environment for evaluation (uses first env from training if None)
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic actions

        Returns:
            Dictionary with evaluation metrics
        """
        # For vectorized envs, use the first env for evaluation
        if eval_env is not None:
            env = eval_env
        elif self.is_vec_env:
            env = self.env.envs[0]
        else:
            env = self.env

        episode_rewards = []
        episode_lengths = []
        episode_costs = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            length = 0

            while not done:
                with torch.no_grad():
                    action, _, _, _ = self.policy(obs, deterministic=deterministic)

                # Pass tensor directly to env.step (no CPU conversion needed)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                length += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            episode_costs.append(-total_reward)  # Cost is negative reward

        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'mean_cost': float(np.mean(episode_costs)),
        }

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        prefix: str = "checkpoint",
    ) -> str:
        """
        Save model checkpoint.

        Args:
            path: Full path for checkpoint (auto-generated if None)
            prefix: Prefix for auto-generated path

        Returns:
            Path to saved checkpoint
        """
        if path is None:
            path = self.checkpoint_dir / f"{prefix}_{self.timesteps_collected}.pt"
        else:
            path = Path(path)

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps_collected': self.timesteps_collected,
            'episodes_completed': self.episodes_completed,
            'updates_performed': self.updates_performed,
            'config': {
                'ppo': self.config.__dict__,
                'training': self.training_config.__dict__,
            },
            'metrics_history': self.metrics_history[-100:],  # Last 100 entries
            'saved_at': datetime.now().isoformat(),
            'tb_log_dir': self.tb_log_dir,
            'log_file': self.log_file,
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        return str(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        env: Union[EpidemicGymEnv, VecEnv],
        policy: GNNActorCritic,
        device: Optional[torch.device] = None,
        tb_log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> 'PPOTrainer':
        """
        Load trainer from checkpoint.

        Args:
            path: Path to checkpoint file
            env: Environment instance (single or vectorized)
            policy: Policy network instance
            device: Torch device
            tb_log_dir: TensorBoard log dir (overrides checkpoint value if provided)
            log_file: JSONL log file (overrides checkpoint value if provided)

        Returns:
            PPOTrainer with restored state
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)

        # Reconstruct config
        config = PPOConfig(**checkpoint['config']['ppo'])
        training_config = TrainingConfig(**checkpoint['config']['training'])

        # Get TensorBoard log directory (CLI arg overrides checkpoint value)
        if tb_log_dir is None:
            tb_log_dir = checkpoint.get('tb_log_dir', None)

        # Get JSONL log file (CLI arg overrides checkpoint value)
        if log_file is None:
            log_file = checkpoint.get('log_file', None)

        # Create trainer with same TensorBoard directory and log file
        trainer = cls(
            env=env,
            policy=policy,
            config=config,
            training_config=training_config,
            device=device,
            tb_log_dir=tb_log_dir,
            log_file=log_file,
        )

        # Restore state
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.timesteps_collected = checkpoint['timesteps_collected']
        trainer.episodes_completed = checkpoint['episodes_completed']
        trainer.updates_performed = checkpoint['updates_performed']
        trainer.metrics_history = checkpoint.get('metrics_history', [])

        if 'lr_scheduler_state_dict' in checkpoint and trainer.lr_scheduler is not None:
            trainer.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        print(f"Resumed from checkpoint: {path}")
        print(f"  Timesteps: {trainer.timesteps_collected}, Episodes: {trainer.episodes_completed}")
        if tb_log_dir:
            print(f"  TensorBoard: {tb_log_dir}")
        if log_file:
            print(f"  Log file: {log_file}")

        return trainer

    def _print_progress(self, metrics: Dict[str, float]) -> None:
        """Print training progress."""
        elapsed = metrics['time_elapsed']
        fps = self.timesteps_collected / elapsed if elapsed > 0 else 0

        print(
            f"Steps: {self.timesteps_collected:>8d} | "
            f"Episodes: {self.episodes_completed:>4d} | "
            f"Reward: {metrics.get('mean_reward', 0):>8.1f} | "
            f"Policy Loss: {metrics['policy_loss']:>7.4f} | "
            f"Value Loss: {metrics['value_loss']:>7.4f} | "
            f"Entropy: {metrics['entropy']:>6.4f} | "
            f"FPS: {fps:>6.0f}"
        )

    def _log_to_tensorboard(self, metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard for live visualization."""
        if self.writer is None:
            return

        step = self.timesteps_collected

        # Losses
        self.writer.add_scalar('losses/policy_loss', metrics['policy_loss'], step)
        self.writer.add_scalar('losses/value_loss', metrics['value_loss'], step)
        self.writer.add_scalar('losses/total_loss', metrics['total_loss'], step)

        # Policy metrics
        self.writer.add_scalar('policy/entropy', metrics['entropy'], step)
        self.writer.add_scalar('policy/approx_kl', metrics['approx_kl'], step)
        self.writer.add_scalar('policy/clip_fraction', metrics['clip_fraction'], step)
        self.writer.add_scalar('policy/learning_rate', metrics['learning_rate'], step)

        # Episode metrics (if available)
        if metrics.get('mean_reward') is not None and metrics.get('num_episodes', 0) > 0:
            self.writer.add_scalar('episode/mean_reward', metrics['mean_reward'], step)
            self.writer.add_scalar('episode/std_reward', metrics.get('std_reward', 0), step)
            self.writer.add_scalar('episode/mean_length', metrics.get('mean_length', 0), step)

        # Cost components
        if 'mean_infection_cost' in metrics:
            self.writer.add_scalar('costs/infection', metrics['mean_infection_cost'], step)
            self.writer.add_scalar('costs/vaccination', metrics['mean_vaccination_cost'], step)
            self.writer.add_scalar('costs/operation', metrics['mean_operation_cost'], step)
            self.writer.add_scalar('costs/transition', metrics['mean_transition_cost'], step)

        # Compartment fractions (S, I, V)
        if 'mean_susceptible_frac' in metrics:
            self.writer.add_scalar('compartments/susceptible', metrics['mean_susceptible_frac'], step)
            self.writer.add_scalar('compartments/infected', metrics['mean_infected_frac'], step)
            self.writer.add_scalar('compartments/vaccinated', metrics['mean_vaccinated_frac'], step)

        # Facility status
        if 'mean_open_facility_frac' in metrics:
            self.writer.add_scalar('facilities/open_fraction', metrics['mean_open_facility_frac'], step)

        # Training progress
        elapsed = metrics.get('time_elapsed', 1)
        fps = step / elapsed if elapsed > 0 else 0
        self.writer.add_scalar('time/fps', fps, step)
        self.writer.add_scalar('time/episodes', self.episodes_completed, step)

        # Flush to ensure data is written
        self.writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer and cleanup resources."""
        if self.writer is not None:
            self.writer.close()

    def get_metrics_dataframe(self):
        """
        Get metrics history as pandas DataFrame.

        Returns:
            DataFrame with training metrics
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.metrics_history)
        except ImportError:
            return self.metrics_history
