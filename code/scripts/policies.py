"""
Policy abstraction for evaluation.

This module provides a unified interface for all policies (learned and static)
used in evaluation. The design supports easy addition of new strategies and
RL methods.

Usage:
    from policies import STATIC_POLICIES, PPOPolicy

    # Get all static baselines
    baselines = [cls() for cls in STATIC_POLICIES.values()]

    # Load a trained PPO policy
    ppo = PPOPolicy.from_checkpoint('checkpoints/final.pt', device)
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import GNNActorCritic


class Policy(ABC):
    """Abstract base class for all policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this policy."""
        pass

    def reset(self, obs: Dict[str, Any]) -> None:
        """Called at the start of each episode.

        Override this for policies that need to track state across steps
        (e.g., FixedInitialPolicy needs to record initial facility status).

        Args:
            obs: Initial observation from environment
        """
        pass

    @abstractmethod
    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True
    ) -> np.ndarray:
        """Select action given observation.

        Args:
            obs: Observation dictionary from EpidemicGymEnv
            deterministic: If True, use deterministic action selection

        Returns:
            Action array of shape [M] with binary facility decisions
        """
        pass


class StaticPolicy(Policy):
    """Base class for rule-based static policies."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class AllOpenPolicy(StaticPolicy):
    """Keep all facilities open at all times."""

    def __init__(self):
        super().__init__("all_open")

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True
    ) -> np.ndarray:
        M = obs["num_facilities"]
        return np.ones(M, dtype=np.int32)


class AllClosedPolicy(StaticPolicy):
    """Keep all facilities closed at all times."""

    def __init__(self):
        super().__init__("all_closed")

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True
    ) -> np.ndarray:
        M = obs["num_facilities"]
        return np.zeros(M, dtype=np.int32)


class RandomPolicy(StaticPolicy):
    """Random facility decisions at each step."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__("random")
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True
    ) -> np.ndarray:
        M = obs["num_facilities"]
        return self.rng.integers(0, 2, size=M, dtype=np.int32)


class FixedInitialPolicy(StaticPolicy):
    """Maintain initial facility configuration throughout episode."""

    def __init__(self):
        super().__init__("fixed_initial")
        self._initial_status = None

    def reset(self, obs: Dict[str, Any]) -> None:
        # Store initial facility status (first column of facility features)
        facility_features = obs["facility_features"]
        if isinstance(facility_features, torch.Tensor):
            self._initial_status = (facility_features[:, 0] > 0.5).cpu().numpy().astype(np.int32)
        else:
            self._initial_status = (facility_features[:, 0] > 0.5).astype(np.int32)

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True
    ) -> np.ndarray:
        if self._initial_status is None:
            raise RuntimeError("FixedInitialPolicy.reset() must be called before select_action()")
        return self._initial_status.copy()


# Registry of all static policies for easy access
STATIC_POLICIES: Dict[str, type] = {
    "all_open": AllOpenPolicy,
    "all_closed": AllClosedPolicy,
    "random": RandomPolicy,
    "fixed_initial": FixedInitialPolicy,
}


class LearnedPolicy(Policy):
    """Base class for learned policies loaded from checkpoints."""

    def __init__(self, name: str, device: torch.device):
        self._name = name
        self.device = device

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: torch.device
    ) -> "LearnedPolicy":
        """Load policy from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model onto

        Returns:
            Loaded policy instance
        """
        pass


class PPOPolicy(LearnedPolicy):
    """PPO policy using GNNActorCritic network."""

    def __init__(
        self,
        name: str,
        policy_network: GNNActorCritic,
        device: torch.device
    ):
        super().__init__(name, device)
        self.policy = policy_network
        self.policy.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: torch.device
    ) -> "PPOPolicy":
        """Load PPO policy from checkpoint.

        Args:
            checkpoint_path: Path to PPO checkpoint (.pt file)
            device: Device to load model onto

        Returns:
            PPOPolicy instance with loaded weights
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config
        ppo_config = checkpoint["config"]["ppo"]

        # Create network with architecture from checkpoint
        policy = GNNActorCritic(
            individual_in_dim=5,  # [S, I, V, x, y]
            facility_in_dim=3,    # [open_status, x, y]
            hidden_dim=ppo_config.get("hidden_dim", 128),
            num_gnn_layers=ppo_config.get("num_gnn_layers", 3),
        ).to(device)

        # Load weights
        policy.load_state_dict(checkpoint["policy_state_dict"])

        # Create name from checkpoint filename
        name = checkpoint_path.stem

        return cls(name, policy, device)

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True
    ) -> np.ndarray:
        """Select action using GNN policy.

        Args:
            obs: Observation dictionary from EpidemicGymEnv
            deterministic: If True, use argmax; otherwise sample

        Returns:
            Action array of shape [M]
        """
        with torch.no_grad():
            action, _, _, _ = self.policy(obs, deterministic=deterministic)

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            return action.cpu().numpy().astype(np.int32)
        return action.astype(np.int32)


# Registry of learned policy loaders (for future RL methods)
LEARNED_POLICY_LOADERS: Dict[str, type] = {
    "ppo": PPOPolicy,
    # Future: "dqn": DQNPolicy, "a2c": A2CPolicy, etc.
}


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    policy_type: str = "ppo"
) -> LearnedPolicy:
    """Load a learned policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load onto
        policy_type: Type of policy ("ppo", etc.)

    Returns:
        Loaded LearnedPolicy instance

    Raises:
        ValueError: If policy_type is not recognized
    """
    if policy_type not in LEARNED_POLICY_LOADERS:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Available: {list(LEARNED_POLICY_LOADERS.keys())}"
        )

    loader_cls = LEARNED_POLICY_LOADERS[policy_type]
    return loader_cls.from_checkpoint(checkpoint_path, device)
