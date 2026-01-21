"""
Gymnasium environment wrapper for EpidemicEnvironment.

This module provides a gym.Env interface for the epidemic simulation,
enabling RL algorithms to interact with the environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, Union

from params import (
    env_params as default_env_params,
    env_params_range as default_env_params_range,
    device as default_device,
)
from funcs import EpidemicEnvironment


class EpidemicGymEnv(gym.Env):
    """
    Gymnasium wrapper for EpidemicEnvironment.

    This environment models epidemic spread on contact networks with
    vaccination facilities. The agent decides which facilities to open
    or close at each decision step.

    Observation Space (Dict):
        - individual_features: [N, 5] - [S, I, V, x, y] for each individual
        - facility_features: [M, 3] - [open_status, x, y] for each facility
        - edge_index_ii: [2, E_ii] - contact network edges
        - edge_index_if: [2, N*M] - individual-facility edges
        - edge_attr_if: [N*M, 1] - distances for individual-facility edges
        - edge_index_ff: [2, E_ff] - facility-facility edges
        - edge_attr_ff: [E_ff, 1] - distances for facility-facility edges
        - global_features: [8] - normalized global state

    Action Space:
        MultiBinary(M) - binary open/close decision for each facility

    Reward:
        Negative total cost per decision step:
        -( f_plus * openings + f_minus * closings +
           C_O * open_facilities + C_I * infections + C_V * vaccinations )
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        randomize: bool = True,
        decision_interval: int = 100,
        max_episode_steps: int = 30,
        terminate_on_resolution: bool = True,
        env_params: Optional[Dict] = None,
        env_params_range: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Gymnasium environment.

        Args:
            randomize: Whether to randomize graph parameters each episode
            decision_interval: Number of dt timesteps between RL decisions
            max_episode_steps: Maximum number of decisions per episode
            terminate_on_resolution: End episode when infected count = 0
            env_params: Base environment parameters (uses defaults if None)
            env_params_range: Ranges for randomization (uses defaults if None)
            device: Torch device for tensors
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()

        self.randomize = randomize
        self.decision_interval = decision_interval
        self.max_episode_steps = max_episode_steps
        self.terminate_on_resolution = terminate_on_resolution
        self.env_params = env_params if env_params is not None else default_env_params.copy()
        self.env_params_range = env_params_range if env_params_range is not None else default_env_params_range.copy()
        self.device = device if device is not None else default_device
        self.render_mode = render_mode

        # Create initial environment to get dimensions
        self._create_epidemic_env()

        # Define action space (will be updated in reset if M changes)
        self._M = self.epidemic_env.M
        self.action_space = spaces.MultiBinary(self._M)

        # Define observation space as Dict
        # We use flexible bounds since graph size can vary
        self.observation_space = self._create_observation_space()

        # Episode state
        self._current_step = 0
        self._cumulative_cost = 0.0
        self._episode_info = {}

    def _create_epidemic_env(self) -> None:
        """Create or recreate the underlying EpidemicEnvironment."""
        self.epidemic_env = EpidemicEnvironment(
            randomize=self.randomize,
            env_params=self.env_params,
            env_params_range=self.env_params_range,
        )

    def _create_observation_space(self) -> spaces.Dict:
        """
        Create the observation space dictionary.

        Uses large bounds to accommodate variable graph sizes.
        """
        max_N = 25000  # Maximum individuals
        max_M = 100    # Maximum facilities
        max_edges_ii = max_N * 50  # Generous edge bound
        max_edges_if = max_N * max_M

        return spaces.Dict({
            "individual_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_N, 5), dtype=np.float32
            ),
            "facility_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_M, 3), dtype=np.float32
            ),
            "edge_index_ii": spaces.Box(
                low=0, high=max_N,
                shape=(2, max_edges_ii), dtype=np.int64
            ),
            "edge_index_if": spaces.Box(
                low=0, high=max(max_N, max_M),
                shape=(2, max_edges_if), dtype=np.int64
            ),
            "edge_attr_if": spaces.Box(
                low=0, high=np.sqrt(2),  # Max distance in unit square
                shape=(max_edges_if, 1), dtype=np.float32
            ),
            "edge_index_ff": spaces.Box(
                low=0, high=max_M,
                shape=(2, max_M * max_M), dtype=np.int64
            ),
            "edge_attr_ff": spaces.Box(
                low=0, high=np.sqrt(2),
                shape=(max_M * max_M, 1), dtype=np.float32
            ),
            "global_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(8,), dtype=np.float32
            ),
            "problem_params": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,), dtype=np.float32  # 12 problem parameters
            ),
            "num_individuals": spaces.Discrete(max_N + 1),
            "num_facilities": spaces.Discrete(max_M + 1),
        })

    def _get_observation(self) -> Dict[str, Any]:
        """
        Extract current state as observation dictionary.

        Returns:
            Dictionary containing graph features and global state.
            All tensor values are kept on GPU for efficiency.
        """
        data = self.epidemic_env.data
        N = self.epidemic_env.N
        M = self.epidemic_env.M

        # Node features - keep on GPU
        individual_features = data['individual'].x
        facility_features = data['facility'].x

        # Edge indices and attributes - keep on GPU
        edge_index_ii = data['individual', 'interacts', 'individual'].edge_index

        edge_index_if = data['individual', 'visits', 'facility'].edge_index
        edge_attr_if = data['individual', 'visits', 'facility'].edge_attr

        edge_index_ff = data['facility', 'connects', 'facility'].edge_index
        edge_attr_ff = data['facility', 'connects', 'facility'].edge_attr

        # Compute global features on GPU
        S_count = individual_features[:, 0].sum()
        I_count = individual_features[:, 1].sum()
        V_count = individual_features[:, 2].sum()
        open_count = facility_features[:, 0].sum()

        # Mean distance to open facility (on GPU)
        min_dists = self.epidemic_env.min_dists_to_open_facilities
        finite_mask = min_dists < float('inf')
        if finite_mask.any():
            mean_dist = min_dists[finite_mask].mean()
        else:
            mean_dist = torch.tensor(1.0, device=self.device)

        global_features = torch.tensor([
            self._current_step / self.max_episode_steps,  # Normalized time
            S_count / N,                                   # Susceptible fraction
            I_count / N,                                   # Infected fraction
            V_count / N,                                   # Vaccinated fraction
            open_count / M,                                # Open facility fraction
            mean_dist,                                     # Mean distance to facility
            N / 20000,                                     # Normalized N
            M / 50,                                        # Normalized M
        ], dtype=torch.float32, device=self.device)

        # Problem parameters (normalized for network stability)
        # These help the policy adapt to different problem instances
        env = self.epidemic_env
        problem_params = torch.tensor([
            # Epidemic dynamics parameters
            float(env.beta_1) / 1.0,      # Infection rate (typical range 0-1)
            float(env.beta_2) / 1.0,      # Breakthrough infection rate
            float(env.delta) / 1.0,       # Recovery rate
            float(env.omega) / 1.0,       # Waning immunity rate
            # Vaccination parameters
            float(env.v_max) / 1.0,       # Max vaccination rate
            float(env.v_min) / 0.1,       # Min vaccination rate
            float(env.alpha) / 100.0,     # Distance decay (typical ~10-100)
            # Cost parameters (log-scaled for stability, using math.log1p to avoid tensor overhead)
            math.log1p(float(env.C_I)) / 10.0,      # Infection cost
            math.log1p(float(env.C_V)) / 10.0,      # Vaccination cost
            math.log1p(float(env.C_O)) / 10.0,      # Operational cost
            math.log1p(float(env.f_plus)) / 10.0,   # Opening cost
            math.log1p(float(env.f_minus)) / 10.0,  # Closing cost
        ], dtype=torch.float32, device=self.device)

        return {
            "individual_features": individual_features,
            "facility_features": facility_features,
            "edge_index_ii": edge_index_ii,
            "edge_index_if": edge_index_if,
            "edge_attr_if": edge_attr_if,
            "edge_index_ff": edge_index_ff,
            "edge_attr_ff": edge_attr_ff,
            "global_features": global_features,
            "problem_params": problem_params,
            "num_individuals": N,
            "num_facilities": M,
        }

    def _compute_transition_cost(
        self,
        old_status: torch.Tensor,
        new_status: torch.Tensor
    ) -> float:
        """
        Compute the cost of facility status transitions.

        Args:
            old_status: Previous facility open/close status [M]
            new_status: New facility open/close status [M]

        Returns:
            Transition cost (f_plus * openings + f_minus * closings)
        """
        openings = ((old_status == 0) & (new_status == 1)).sum().item()
        closings = ((old_status == 1) & (new_status == 0)).sum().item()

        f_plus = float(self.epidemic_env.f_plus)
        f_minus = float(self.epidemic_env.f_minus)

        return f_plus * openings + f_minus * closings

    def _update_facility_status(self, action: torch.Tensor) -> torch.Tensor:
        """
        Update facility open/close status based on action.

        Args:
            action: Binary action tensor [M] on GPU

        Returns:
            Old facility status tensor
        """
        # Get old status
        old_status = self.epidemic_env.data['facility'].x[:, 0].clone()

        # Update status (action is already a tensor on GPU)
        self.epidemic_env.data['facility'].x[:, 0] = action

        # Recompute vaccination distances
        self.epidemic_env.min_dists_to_open_facilities = \
            self.epidemic_env._compute_min_dists_to_open_facilities()

        return old_status

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create new environment (with randomization if enabled)
        self._create_epidemic_env()

        # Update action space if M changed
        new_M = self.epidemic_env.M
        if new_M != self._M:
            self._M = new_M
            self.action_space = spaces.MultiBinary(self._M)

        # Reset episode state
        self._current_step = 0
        self._cumulative_cost = 0.0
        self._episode_info = {
            "total_infections": 0,
            "total_vaccinations": 0,
            "facility_changes": 0,
            "N": self.epidemic_env.N,
            "M": self.epidemic_env.M,
        }

        obs = self._get_observation()
        info = {"episode_info": self._episode_info.copy()}

        return obs, info

    def step(
        self,
        action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one decision step in the environment.

        Args:
            action: Binary array/tensor of length M (1=open, 0=closed)
                    Accepts both numpy arrays and torch tensors for efficiency.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)

        Note on reward scaling:
            To ensure consistent reward magnitudes across different problem sizes,
            costs are scaled:
            - Individual-related costs (infections, vaccinations) are scaled by N
            - Facility-related costs (operations, transitions) are scaled by M
            This produces rewards roughly in the range [-10, 0] regardless of N and M.
        """
        # Convert action to tensor if needed (stay on GPU if already tensor)
        if isinstance(action, torch.Tensor):
            action_tensor = action.to(device=self.device, dtype=torch.float32).flatten()
        else:
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).flatten()

        if action_tensor.shape[0] != self._M:
            raise ValueError(f"Action length {action_tensor.shape[0]} != M={self._M}")

        # Get N and M for scaling
        N = self.epidemic_env.N
        M = self.epidemic_env.M

        # 1. Compute transition costs (facility-related, scale by M)
        old_status = self.epidemic_env.data['facility'].x[:, 0].clone()
        transition_cost = self._compute_transition_cost(old_status, action_tensor)

        # Track facility changes (compare on GPU)
        changes = (old_status != action_tensor).sum().item()
        self._episode_info["facility_changes"] += int(changes)

        # 2. Update facility status
        self._update_facility_status(action_tensor)

        # 3. Run epidemic simulation using optimized GPU step function
        _, _, details = self.epidemic_env.step(
            num_dt=self.decision_interval,
            verbose=False,
            visualize=False,
            return_details=True
        )

        # Extract costs from simulation
        individual_cost = details['individual_cost']
        facility_cost = details['facility_cost']
        infection_cost = details.get('infection_cost', 0)
        vaccination_cost = details.get('vaccination_cost', 0)
        operation_cost = details.get('operation_cost', facility_cost)
        current_I = details['current_infected']
        current_S = details.get('current_susceptible', 0)
        current_V = details.get('current_vaccinated', 0)

        # Update episode tracking
        self._episode_info["total_infections"] += details['total_infections']
        self._episode_info["total_vaccinations"] += details['total_vaccinations']

        # 4. Total cost for this decision step (unscaled, for tracking)
        total_cost = transition_cost + individual_cost + facility_cost
        self._cumulative_cost += total_cost

        # 5. Compute reward with proper scaling
        # Scale individual costs by N, facility costs by M
        scaled_individual_cost = individual_cost / N
        scaled_facility_cost = (transition_cost + facility_cost) / M
        scaled_total_cost = scaled_individual_cost + scaled_facility_cost
        reward = -scaled_total_cost

        # 6. Check termination conditions
        self._current_step += 1

        # Terminated: epidemic resolved (no infections)
        terminated = self.terminate_on_resolution and (current_I == 0)

        # Truncated: max steps reached
        truncated = self._current_step >= self.max_episode_steps

        # 7. Get observation and info
        obs = self._get_observation()
        info = {
            "episode_info": self._episode_info.copy(),
            "step_cost": total_cost,
            "scaled_step_cost": scaled_total_cost,
            # Cost components
            "infection_cost": infection_cost,
            "vaccination_cost": vaccination_cost,
            "operation_cost": operation_cost,
            "transition_cost": transition_cost,
            "individual_cost": individual_cost,
            "facility_cost": facility_cost,
            "cumulative_cost": self._cumulative_cost,
            # Compartment counts
            "current_susceptible": current_S,
            "current_infected": current_I,
            "current_vaccinated": current_V,
            # Step info
            "current_step": self._current_step,
            "open_facilities": int(action_tensor.sum().item()),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.epidemic_env.visualize_network(legend=True, save=False)
        elif self.render_mode == "rgb_array":
            # Would need to capture figure as array
            pass

    def close(self):
        """Clean up resources."""
        pass

    def get_epidemic_env(self) -> EpidemicEnvironment:
        """Get the underlying EpidemicEnvironment instance."""
        return self.epidemic_env

    @classmethod
    def from_saved_problem(
        cls,
        filepath: str,
        decision_interval: int = 100,
        max_episode_steps: int = 30,
        terminate_on_resolution: bool = True,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> "EpidemicGymEnv":
        """
        Create environment from a saved problem file.

        Args:
            filepath: Path to saved problem (.pt file)
            decision_interval: Number of dt timesteps between decisions
            max_episode_steps: Maximum decisions per episode
            terminate_on_resolution: End on epidemic resolution
            device: Torch device
            **kwargs: Additional arguments

        Returns:
            EpidemicGymEnv initialized with the saved problem
        """
        # Create env with randomize=False (we'll load the problem)
        env = cls(
            randomize=False,
            decision_interval=decision_interval,
            max_episode_steps=max_episode_steps,
            terminate_on_resolution=terminate_on_resolution,
            device=device,
            **kwargs
        )

        # Load the saved problem
        env.epidemic_env = EpidemicEnvironment.load(filepath, device=device)

        # Update action space
        env._M = env.epidemic_env.M
        env.action_space = spaces.MultiBinary(env._M)

        return env
