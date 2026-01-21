"""
Rollout buffer for PPO training.

Stores trajectories and computes Generalized Advantage Estimation (GAE).
Handles variable-size graph observations.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Iterator, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition in rollout buffer."""
    observation: Dict[str, Any]           # Graph observation dictionary
    action: Union[Tensor, np.ndarray]     # Binary actions [M] (tensor or numpy)
    log_prob: float                       # Sum of action log probabilities
    reward: float                         # Scalar reward
    value: float                          # Value estimate
    done: bool                            # Episode termination flag


class RolloutBuffer:
    """
    Buffer for storing PPO rollout trajectories.

    Handles variable-size graphs by storing observation dictionaries.
    Computes advantages using Generalized Advantage Estimation (GAE).

    GAE Formula:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize the rollout buffer.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage
        self.observations: List[Dict[str, Any]] = []
        self.actions: List[Union[Tensor, np.ndarray]] = []  # Can be tensor or numpy
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

        # Computed after rollout
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

        # For vectorized environments
        self.env_indices: List[int] = []

    def add(
        self,
        obs: Dict[str, Any],
        action: Union[Tensor, np.ndarray],
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        env_idx: int = 0,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: Observation dictionary
            action: Action taken [M] (tensor or numpy array)
            log_prob: Log probability of action (sum over facilities)
            reward: Reward received
            value: Value estimate V(s)
            done: Episode done flag
            env_idx: Environment index (for vectorized envs)
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.env_indices.append(env_idx)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
    ) -> None:
        """
        Compute GAE advantages and returns.

        Uses the formula:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}

        Args:
            last_value: Value estimate for the final state
            last_done: Whether the final state is terminal
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        # Convert to arrays for efficient computation
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Append last value for bootstrapping
        values_ext = np.append(values, last_value)
        dones_ext = np.append(dones, float(last_done))

        # Compute GAE backwards
        gae = 0.0
        for t in reversed(range(n)):
            # TD error
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones_ext[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_ext[t]) * gae
            self.advantages[t] = gae

        # Returns = advantages + values
        self.returns = self.advantages + values

    def compute_returns_and_advantages_vec(
        self,
        last_values: np.ndarray,
        last_dones: List[bool],
    ) -> None:
        """
        Compute GAE advantages for vectorized environments.

        Handles multiple environments where each may have different
        episode boundaries. The buffer stores transitions interleaved
        from all environments.

        Args:
            last_values: Value estimates for final states [num_envs]
            last_dones: Whether final states are terminal [num_envs]
        """
        n = len(self.rewards)
        if n == 0:
            self.advantages = np.array([], dtype=np.float32)
            self.returns = np.array([], dtype=np.float32)
            return

        num_envs = len(last_values)

        # Convert to arrays
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        env_indices = np.array(self.env_indices, dtype=np.int32)

        self.advantages = np.zeros(n, dtype=np.float32)

        # Compute GAE backwards, tracking per-env state
        # gae[env_idx] tracks the running GAE for each env
        gae = np.zeros(num_envs, dtype=np.float32)

        # For each env, find the last transition and set up bootstrap
        for env_idx in range(num_envs):
            gae[env_idx] = 0.0  # Will be updated with bootstrap

        # Process backwards
        for t in reversed(range(n)):
            env_idx = env_indices[t]

            # Determine next value for bootstrapping
            # Find next transition for this env, or use last_value
            next_idx = None
            for future_t in range(t + 1, n):
                if env_indices[future_t] == env_idx:
                    next_idx = future_t
                    break

            if next_idx is not None:
                next_value = values[next_idx]
                next_done = dones[t]  # Use current done flag (transition boundary)
            else:
                # This is the last transition for this env - use bootstrap
                next_value = last_values[env_idx]
                next_done = float(last_dones[env_idx])

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]

            # GAE - reset if done, otherwise propagate
            if dones[t]:
                gae[env_idx] = delta
            else:
                gae[env_idx] = delta + self.gamma * self.gae_lambda * gae[env_idx]

            self.advantages[t] = gae[env_idx]

        # Returns = advantages + values
        self.returns = self.advantages + values

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield mini-batches for PPO updates.

        Due to variable graph sizes, each batch contains indices
        that must be processed individually.

        Args:
            batch_size: Number of transitions per batch
            shuffle: Whether to shuffle indices

        Yields:
            Dictionary with batch data
        """
        n = len(self.rewards)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        # Generate batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                'indices': batch_indices,
                'observations': [self.observations[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],  # List for variable-size M
                'old_log_probs': np.array([self.log_probs[i] for i in batch_indices]),
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
                'values': np.array([self.values[i] for i in batch_indices]),
            }

    def get_all(self) -> Dict[str, Any]:
        """
        Get all data in buffer.

        Returns:
            Dictionary with all transitions
        """
        return {
            'observations': self.observations,
            'actions': self.actions,  # Keep as list (may contain tensors)
            'old_log_probs': np.array(self.log_probs),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'dones': np.array(self.dones),
            'advantages': self.advantages,
            'returns': self.returns,
        }

    def clear(self) -> None:
        """Clear buffer for new rollout."""
        self.observations: List[Dict[str, Any]] = []
        self.actions: List[Union[Tensor, np.ndarray]] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.env_indices: List[int] = []
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.rewards)

    @property
    def size(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.rewards)


class EpisodeBuffer:
    """
    Buffer for tracking episode-level statistics.

    Useful for logging episode returns and lengths.
    """

    def __init__(self, max_episodes: int = 100):
        """
        Initialize episode buffer.

        Args:
            max_episodes: Maximum number of episodes to track
        """
        self.max_episodes = max_episodes
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_infos: List[Dict] = []

    def add_episode(
        self,
        total_reward: float,
        length: int,
        info: Optional[Dict] = None,
    ) -> None:
        """
        Record a completed episode.

        Args:
            total_reward: Total episode reward
            length: Episode length
            info: Additional episode info
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_infos.append(info if info is not None else {})

        # Keep only recent episodes
        if len(self.episode_rewards) > self.max_episodes:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.episode_infos.pop(0)

    def get_stats(self) -> Dict[str, float]:
        """
        Get episode statistics.

        Returns:
            Dictionary with mean, std, min, max for rewards and lengths
        """
        if len(self.episode_rewards) == 0:
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'mean_length': 0.0,
                'num_episodes': 0,
            }

        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        return {
            'mean_reward': float(rewards.mean()),
            'std_reward': float(rewards.std()),
            'min_reward': float(rewards.min()),
            'max_reward': float(rewards.max()),
            'mean_length': float(lengths.mean()),
            'num_episodes': len(rewards),
        }

    def clear(self) -> None:
        """Clear all episode records."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = []


class StepMetricsBuffer:
    """
    Buffer for tracking step-level metrics for TensorBoard logging.

    Tracks cost components, compartment counts, and facility status.
    """

    def __init__(self, max_steps: int = 10000):
        """
        Initialize step metrics buffer.

        Args:
            max_steps: Maximum number of steps to track
        """
        self.max_steps = max_steps

        # Cost components
        self.infection_costs: List[float] = []
        self.vaccination_costs: List[float] = []
        self.operation_costs: List[float] = []
        self.transition_costs: List[float] = []

        # Compartment counts (as fractions)
        self.susceptible_fracs: List[float] = []
        self.infected_fracs: List[float] = []
        self.vaccinated_fracs: List[float] = []

        # Facility info
        self.open_facility_fracs: List[float] = []

    def add_step(
        self,
        info: Dict[str, Any],
        N: int,
        M: int,
    ) -> None:
        """
        Record step metrics.

        Args:
            info: Info dict from env.step()
            N: Number of individuals
            M: Number of facilities
        """
        # Cost components
        self.infection_costs.append(info.get('infection_cost', 0))
        self.vaccination_costs.append(info.get('vaccination_cost', 0))
        self.operation_costs.append(info.get('operation_cost', 0))
        self.transition_costs.append(info.get('transition_cost', 0))

        # Compartment fractions
        current_S = info.get('current_susceptible', 0)
        current_I = info.get('current_infected', 0)
        current_V = info.get('current_vaccinated', 0)

        self.susceptible_fracs.append(current_S / N if N > 0 else 0)
        self.infected_fracs.append(current_I / N if N > 0 else 0)
        self.vaccinated_fracs.append(current_V / N if N > 0 else 0)

        # Facility status
        open_facs = info.get('open_facilities', 0)
        self.open_facility_fracs.append(open_facs / M if M > 0 else 0)

        # Trim if too large
        if len(self.infection_costs) > self.max_steps:
            self._trim()

    def _trim(self) -> None:
        """Keep only recent steps."""
        trim_size = self.max_steps // 2
        self.infection_costs = self.infection_costs[-trim_size:]
        self.vaccination_costs = self.vaccination_costs[-trim_size:]
        self.operation_costs = self.operation_costs[-trim_size:]
        self.transition_costs = self.transition_costs[-trim_size:]
        self.susceptible_fracs = self.susceptible_fracs[-trim_size:]
        self.infected_fracs = self.infected_fracs[-trim_size:]
        self.vaccinated_fracs = self.vaccinated_fracs[-trim_size:]
        self.open_facility_fracs = self.open_facility_fracs[-trim_size:]

    def get_stats(self) -> Dict[str, float]:
        """
        Get step statistics.

        Returns:
            Dictionary with mean values for all tracked metrics
        """
        if len(self.infection_costs) == 0:
            return {}

        return {
            # Cost components (mean per step)
            'mean_infection_cost': float(np.mean(self.infection_costs)),
            'mean_vaccination_cost': float(np.mean(self.vaccination_costs)),
            'mean_operation_cost': float(np.mean(self.operation_costs)),
            'mean_transition_cost': float(np.mean(self.transition_costs)),
            # Compartment fractions
            'mean_susceptible_frac': float(np.mean(self.susceptible_fracs)),
            'mean_infected_frac': float(np.mean(self.infected_fracs)),
            'mean_vaccinated_frac': float(np.mean(self.vaccinated_fracs)),
            # Facility status
            'mean_open_facility_frac': float(np.mean(self.open_facility_fracs)),
        }

    def clear(self) -> None:
        """Clear all step records."""
        self.infection_costs = []
        self.vaccination_costs = []
        self.operation_costs = []
        self.transition_costs = []
        self.susceptible_fracs = []
        self.infected_fracs = []
        self.vaccinated_fracs = []
        self.open_facility_fracs = []
