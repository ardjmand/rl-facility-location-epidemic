"""
Vectorized environment wrapper for parallel rollout collection.

Manages multiple EpidemicGymEnv instances running in parallel using
ThreadPoolExecutor for concurrent environment stepping.

Supports sequential mode and limited concurrency to prevent GPU overheating.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from torch import Tensor

from .gym_env import EpidemicGymEnv


class VecEnv:
    """
    Vectorized environment that manages multiple EpidemicGymEnv instances.

    Uses ThreadPoolExecutor for parallel environment stepping.
    Handles async resets when individual environments finish episodes.

    Attributes:
        num_envs: Number of parallel environments
        envs: List of EpidemicGymEnv instances
        observations: Current observations from all environments
        sequential: If True, step environments one at a time (prevents GPU overheating)
        max_concurrent: Maximum number of environments to step concurrently
    """

    def __init__(
        self,
        num_envs: int = 4,
        randomize: bool = True,
        decision_interval: int = 100,
        max_episode_steps: int = 30,
        num_workers: Optional[int] = None,
        sequential: bool = False,
        max_concurrent: Optional[int] = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            randomize: Whether to randomize environment parameters
            decision_interval: Steps between RL decisions
            max_episode_steps: Maximum steps per episode
            num_workers: Number of worker threads (defaults to num_envs)
            sequential: If True, step environments sequentially (no parallelism).
                       This prevents GPU overheating but reduces throughput.
            max_concurrent: Maximum environments to step concurrently.
                           If None, uses num_workers. Set to 1 for sequential.
                           Values between 1 and num_envs provide partial parallelism.
        """
        self.num_envs = num_envs
        self.randomize = randomize
        self.decision_interval = decision_interval
        self.max_episode_steps = max_episode_steps
        self.sequential = sequential

        # Determine effective concurrency
        if sequential:
            self.max_concurrent = 1
            self.num_workers = 1
        elif max_concurrent is not None:
            self.max_concurrent = min(max_concurrent, num_envs)
            self.num_workers = self.max_concurrent
        else:
            self.num_workers = num_workers or num_envs
            self.max_concurrent = self.num_workers

        # Create environments
        self.envs: List[EpidemicGymEnv] = []
        for i in range(num_envs):
            env = EpidemicGymEnv(
                randomize=randomize,
                decision_interval=decision_interval,
                max_episode_steps=max_episode_steps,
            )
            self.envs.append(env)

        # Thread pool for parallel execution (only if not fully sequential)
        if self.max_concurrent > 1:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = None

        # Current observations (one per env)
        self.observations: List[Dict[str, Any]] = [None] * num_envs

        # Track which envs need reset
        self._needs_reset = [True] * num_envs

    def reset(self) -> List[Dict[str, Any]]:
        """
        Reset all environments (sequentially or in parallel based on settings).

        Returns:
            List of observations from all environments
        """
        if self.sequential or self.executor is None:
            # Sequential reset
            for i in range(self.num_envs):
                obs, _ = self.envs[i].reset()
                self.observations[i] = obs
                self._needs_reset[i] = False
        else:
            # Parallel reset
            def reset_env(idx: int) -> Tuple[int, Dict[str, Any]]:
                obs, _ = self.envs[idx].reset()
                return idx, obs

            # Submit all resets
            futures = [self.executor.submit(reset_env, i) for i in range(self.num_envs)]

            # Collect results
            for future in as_completed(futures):
                idx, obs = future.result()
                self.observations[idx] = obs
                self._needs_reset[idx] = False

        return self.observations.copy()

    def step(
        self,
        actions: List[Union[np.ndarray, Tensor]],
    ) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        """
        Step all environments (sequentially or in parallel based on settings).

        Args:
            actions: List of actions, one per environment

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"

        # Initialize result lists
        next_observations = [None] * self.num_envs
        rewards = [0.0] * self.num_envs
        terminateds = [False] * self.num_envs
        truncateds = [False] * self.num_envs
        infos = [{}] * self.num_envs

        if self.sequential or self.executor is None:
            # Sequential step
            for idx in range(self.num_envs):
                obs, reward, terminated, truncated, info = self.envs[idx].step(actions[idx])
                next_observations[idx] = obs
                rewards[idx] = reward
                terminateds[idx] = terminated
                truncateds[idx] = truncated
                infos[idx] = info

                # Mark for reset if done
                if terminated or truncated:
                    self._needs_reset[idx] = True
        else:
            # Parallel step
            def step_env(idx: int, action: Union[np.ndarray, Tensor]) -> Tuple[int, Dict[str, Any], float, bool, bool, Dict[str, Any]]:
                obs, reward, terminated, truncated, info = self.envs[idx].step(action)
                return idx, obs, reward, terminated, truncated, info

            # Submit all steps
            futures = [
                self.executor.submit(step_env, i, actions[i])
                for i in range(self.num_envs)
            ]

            # Collect results
            for future in as_completed(futures):
                idx, obs, reward, terminated, truncated, info = future.result()
                next_observations[idx] = obs
                rewards[idx] = reward
                terminateds[idx] = terminated
                truncateds[idx] = truncated
                infos[idx] = info

                # Mark for reset if done
                if terminated or truncated:
                    self._needs_reset[idx] = True

        self.observations = next_observations
        return next_observations, rewards, terminateds, truncateds, infos

    def step_and_reset(
        self,
        actions: List[Union[np.ndarray, Tensor]],
    ) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[bool], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Step all environments and auto-reset finished ones.

        This is more efficient than separate step + reset calls as it
        handles resets in the same parallel execution.

        Args:
            actions: List of actions, one per environment

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos, reset_infos)
            - observations: Next observations (post-reset for done envs)
            - reset_infos: Info dicts from reset for done envs (empty dict if not reset)
        """
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"

        # Initialize result lists
        next_observations = [None] * self.num_envs
        rewards = [0.0] * self.num_envs
        terminateds = [False] * self.num_envs
        truncateds = [False] * self.num_envs
        infos = [{}] * self.num_envs
        reset_infos = [{}] * self.num_envs

        if self.sequential or self.executor is None:
            # Sequential step and reset
            for idx in range(self.num_envs):
                obs, reward, terminated, truncated, info = self.envs[idx].step(actions[idx])
                reset_info = {}

                # Auto-reset if done
                if terminated or truncated:
                    obs, reset_info = self.envs[idx].reset()

                next_observations[idx] = obs
                rewards[idx] = reward
                terminateds[idx] = terminated
                truncateds[idx] = truncated
                infos[idx] = info
                reset_infos[idx] = reset_info
        else:
            # Parallel step and reset
            def step_and_maybe_reset(
                idx: int,
                action: Union[np.ndarray, Tensor]
            ) -> Tuple[int, Dict[str, Any], float, bool, bool, Dict[str, Any], Dict[str, Any]]:
                obs, reward, terminated, truncated, info = self.envs[idx].step(action)
                reset_info = {}

                # Auto-reset if done
                if terminated or truncated:
                    obs, reset_info = self.envs[idx].reset()

                return idx, obs, reward, terminated, truncated, info, reset_info

            # Submit all steps
            futures = [
                self.executor.submit(step_and_maybe_reset, i, actions[i])
                for i in range(self.num_envs)
            ]

            # Collect results
            for future in as_completed(futures):
                idx, obs, reward, terminated, truncated, info, reset_info = future.result()
                next_observations[idx] = obs
                rewards[idx] = reward
                terminateds[idx] = terminated
                truncateds[idx] = truncated
                infos[idx] = info
                reset_infos[idx] = reset_info

        self.observations = next_observations
        return next_observations, rewards, terminateds, truncateds, infos, reset_infos

    def close(self):
        """Clean up resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        for env in self.envs:
            env.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass

    @property
    def single_observation_space(self):
        """Get observation space from first environment."""
        return self.envs[0].observation_space

    @property
    def single_action_space(self):
        """Get action space from first environment."""
        return self.envs[0].action_space
