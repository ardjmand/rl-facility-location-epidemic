"""
GNN-based Actor-Critic networks for epidemic facility location.

This module provides:
- HeteroGNNEncoder: Heterogeneous GNN encoder using HeteroConv with SAGEConv
- ActorHead: Per-facility action logits
- CriticHead: Global state value estimation
- GNNActorCritic: Combined actor-critic with shared backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli
from torch_geometric.nn import SAGEConv, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData, Batch
from torch_scatter import scatter_mean, scatter_add
from typing import Dict, Tuple, Optional, Any, List
import numpy as np


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN encoder for epidemic graphs.

    Processes individual and facility nodes through multiple
    HeteroConv layers with SAGEConv for each edge type.

    Architecture:
        Input projection -> L x (HeteroConv + ReLU + Dropout) -> Output embeddings
    """

    def __init__(
        self,
        individual_in_dim: int = 5,
        facility_in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize the heterogeneous GNN encoder.

        Args:
            individual_in_dim: Input dimension for individual nodes (default 5: S,I,V,x,y)
            facility_in_dim: Input dimension for facility nodes (default 3: open,x,y)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.individual_in_dim = individual_in_dim
        self.facility_in_dim = facility_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projections to common hidden dimension
        self.individual_proj = nn.Linear(individual_in_dim, hidden_dim)
        self.facility_proj = nn.Linear(facility_in_dim, hidden_dim)

        # Build HeteroConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('individual', 'interacts', 'individual'): SAGEConv(
                    hidden_dim, hidden_dim
                ),
                ('individual', 'visits', 'facility'): SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                ),
                ('facility', 'visited_by', 'individual'): SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                ),
                ('facility', 'connects', 'facility'): SAGEConv(
                    hidden_dim, hidden_dim
                ),
            }, aggr='sum')
            self.convs.append(conv)

        # Layer normalization for stability
        self.individual_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.facility_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """
        Forward pass through GNN layers.

        Args:
            x_dict: Node features {'individual': [N, in_dim], 'facility': [M, in_dim]}
            edge_index_dict: Edge indices for each edge type

        Returns:
            Node embeddings {'individual': [N, H], 'facility': [M, H]}
        """
        # Project inputs to hidden dimension
        h_dict = {
            'individual': self.individual_proj(x_dict['individual']),
            'facility': self.facility_proj(x_dict['facility']),
        }

        # Apply GNN layers with residual connections
        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)

            # Apply layer norm and residual connection
            h_dict_new['individual'] = self.individual_norms[i](h_dict_new['individual'])
            h_dict_new['facility'] = self.facility_norms[i](h_dict_new['facility'])

            # Residual connection
            h_dict['individual'] = F.relu(h_dict['individual'] + h_dict_new['individual'])
            h_dict['facility'] = F.relu(h_dict['facility'] + h_dict_new['facility'])

            # Dropout
            h_dict['individual'] = self.dropout_layer(h_dict['individual'])
            h_dict['facility'] = self.dropout_layer(h_dict['facility'])

        return h_dict


class ActorHead(nn.Module):
    """
    Actor head that outputs per-facility action logits.

    Takes facility embeddings and produces independent Bernoulli
    logits for open/close decisions.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        global_feature_dim: int = 8,
        problem_param_dim: int = 12,
    ):
        """
        Initialize the actor head.

        Args:
            hidden_dim: Dimension of facility embeddings
            global_feature_dim: Dimension of global features
            problem_param_dim: Dimension of problem parameters
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.global_feature_dim = global_feature_dim
        self.problem_param_dim = problem_param_dim

        # MLP for facility logits
        # Concatenate facility embedding with global features and problem params
        input_dim = hidden_dim + global_feature_dim + problem_param_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        facility_embeddings: Tensor,
        global_features: Tensor,
        problem_params: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute per-facility logits.

        Args:
            facility_embeddings: [M, H] facility node embeddings
            global_features: [1, D_g] or [D_g] global state features
            problem_params: [1, D_p] or [D_p] problem parameters (optional)

        Returns:
            logits: [M] per-facility open/close logits
        """
        M = facility_embeddings.size(0)

        # Expand global features to match facility count
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        global_features = global_features.expand(M, -1)

        # Handle problem params
        if problem_params is not None:
            if problem_params.dim() == 1:
                problem_params = problem_params.unsqueeze(0)
            problem_params = problem_params.expand(M, -1)
            # Concatenate facility embeddings, global features, and problem params
            combined = torch.cat([facility_embeddings, global_features, problem_params], dim=-1)
        else:
            # Backward compatibility: use zeros if not provided
            device = facility_embeddings.device
            zeros = torch.zeros(M, self.problem_param_dim, device=device)
            combined = torch.cat([facility_embeddings, global_features, zeros], dim=-1)

        logits = self.mlp(combined).squeeze(-1)

        return logits


class CriticHead(nn.Module):
    """
    Critic head that outputs scalar state value.

    Uses global pooling over node embeddings combined with
    global features and problem parameters.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        global_feature_dim: int = 8,
        problem_param_dim: int = 12,
    ):
        """
        Initialize the critic head.

        Args:
            hidden_dim: Dimension of node embeddings
            global_feature_dim: Dimension of global features
            problem_param_dim: Dimension of problem parameters
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.global_feature_dim = global_feature_dim
        self.problem_param_dim = problem_param_dim

        # Input: pooled individual + pooled facility + global features + problem params
        input_dim = hidden_dim * 2 + global_feature_dim + problem_param_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        individual_embeddings: Tensor,
        facility_embeddings: Tensor,
        global_features: Tensor,
        problem_params: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute state value estimate.

        Args:
            individual_embeddings: [N, H] individual node embeddings
            facility_embeddings: [M, H] facility node embeddings
            global_features: [1, D_g] or [D_g] global state features
            problem_params: [1, D_p] or [D_p] problem parameters (optional)

        Returns:
            value: Scalar state value
        """
        # Mean pool over nodes
        individual_pool = individual_embeddings.mean(dim=0)  # [H]
        facility_pool = facility_embeddings.mean(dim=0)      # [H]

        # Ensure global features is 1D
        if global_features.dim() == 2:
            global_features = global_features.squeeze(0)

        # Handle problem params
        if problem_params is not None:
            if problem_params.dim() == 2:
                problem_params = problem_params.squeeze(0)
            # Concatenate all features including problem params
            combined = torch.cat([
                individual_pool,
                facility_pool,
                global_features,
                problem_params
            ], dim=-1)
        else:
            # Backward compatibility: use zeros if not provided
            device = individual_embeddings.device
            zeros = torch.zeros(self.problem_param_dim, device=device)
            combined = torch.cat([
                individual_pool,
                facility_pool,
                global_features,
                zeros
            ], dim=-1)

        # Compute value
        value = self.mlp(combined).squeeze(-1)

        return value


class GNNActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared GNN backbone.

    Architecture:
        Observation -> HeteroGNNEncoder (shared)
                            |
              +-------------+-------------+
              |                           |
         ActorHead                   CriticHead
    (per-facility logits)         (global value)

    Both heads receive problem parameters to condition on
    the specific problem instance (useful for domain randomization).
    """

    def __init__(
        self,
        individual_in_dim: int = 5,
        facility_in_dim: int = 3,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        global_feature_dim: int = 8,
        problem_param_dim: int = 12,
        dropout: float = 0.1,
    ):
        """
        Initialize the actor-critic network.

        Args:
            individual_in_dim: Input dim for individual nodes
            facility_in_dim: Input dim for facility nodes
            hidden_dim: Hidden dimension
            num_gnn_layers: Number of GNN layers
            global_feature_dim: Dimension of global features
            problem_param_dim: Dimension of problem parameters
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.global_feature_dim = global_feature_dim
        self.problem_param_dim = problem_param_dim

        # Shared GNN encoder
        self.encoder = HeteroGNNEncoder(
            individual_in_dim=individual_in_dim,
            facility_in_dim=facility_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Actor head (with problem params)
        self.actor = ActorHead(
            hidden_dim=hidden_dim,
            global_feature_dim=global_feature_dim,
            problem_param_dim=problem_param_dim,
        )

        # Critic head (with problem params)
        self.critic = CriticHead(
            hidden_dim=hidden_dim,
            global_feature_dim=global_feature_dim,
            problem_param_dim=problem_param_dim,
        )

    def _obs_to_tensors(
        self,
        obs: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Dict[str, Tensor], Dict[Tuple, Tensor], Tensor, Optional[Tensor]]:
        """
        Convert observation dictionary to tensors.

        Handles both numpy arrays and GPU tensors as input for efficiency.
        When tensors are already on the correct device, no conversion is needed.

        Args:
            obs: Observation dictionary from environment
            device: Target device

        Returns:
            Tuple of (x_dict, edge_index_dict, global_features, problem_params)
        """
        # Node features - handle both numpy and tensor
        N = obs['num_individuals']
        M = obs['num_facilities']

        ind_feat = obs['individual_features']
        fac_feat = obs['facility_features']

        if isinstance(ind_feat, Tensor):
            x_individual = ind_feat[:N].to(device=device, dtype=torch.float32)
        else:
            x_individual = torch.tensor(ind_feat[:N], dtype=torch.float32, device=device)

        if isinstance(fac_feat, Tensor):
            x_facility = fac_feat[:M].to(device=device, dtype=torch.float32)
        else:
            x_facility = torch.tensor(fac_feat[:M], dtype=torch.float32, device=device)

        x_dict = {
            'individual': x_individual,
            'facility': x_facility,
        }

        # Edge indices - handle both numpy and tensor
        edge_ii = obs['edge_index_ii']
        edge_if = obs['edge_index_if']
        edge_ff = obs['edge_index_ff']

        if isinstance(edge_ii, Tensor):
            edge_index_ii = edge_ii.to(device=device, dtype=torch.long)
        else:
            edge_index_ii = torch.tensor(edge_ii, dtype=torch.long, device=device)

        if isinstance(edge_if, Tensor):
            edge_index_if = edge_if.to(device=device, dtype=torch.long)
            edge_index_fi = edge_if[[1, 0]].to(device=device, dtype=torch.long)
        else:
            edge_index_if = torch.tensor(edge_if, dtype=torch.long, device=device)
            edge_index_fi = torch.tensor(edge_if[[1, 0]], dtype=torch.long, device=device)

        if isinstance(edge_ff, Tensor):
            edge_index_ff = edge_ff.to(device=device, dtype=torch.long)
        else:
            edge_index_ff = torch.tensor(edge_ff, dtype=torch.long, device=device)

        edge_index_dict = {
            ('individual', 'interacts', 'individual'): edge_index_ii,
            ('individual', 'visits', 'facility'): edge_index_if,
            ('facility', 'visited_by', 'individual'): edge_index_fi,
            ('facility', 'connects', 'facility'): edge_index_ff,
        }

        # Global features - handle both numpy and tensor
        glob_feat = obs['global_features']
        if isinstance(glob_feat, Tensor):
            global_features = glob_feat.to(device=device, dtype=torch.float32)
        else:
            global_features = torch.tensor(glob_feat, dtype=torch.float32, device=device)

        # Problem parameters (optional, for backward compatibility)
        problem_params = None
        if 'problem_params' in obs:
            prob_params = obs['problem_params']
            if isinstance(prob_params, Tensor):
                problem_params = prob_params.to(device=device, dtype=torch.float32)
            else:
                problem_params = torch.tensor(prob_params, dtype=torch.float32, device=device)

        return x_dict, edge_index_dict, global_features, problem_params

    def forward(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass producing actions and values.

        Args:
            obs: Observation dictionary from environment
            deterministic: If True, return argmax actions

        Returns:
            Tuple of:
                - actions: Binary action tensor [M]
                - log_probs: Sum of log probabilities (scalar)
                - entropy: Mean entropy (scalar)
                - value: State value estimate (scalar)
        """
        device = next(self.parameters()).device

        # Convert observation to tensors
        x_dict, edge_index_dict, global_features, problem_params = self._obs_to_tensors(obs, device)

        # Encode graph
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get actor logits (with problem params)
        logits = self.actor(h_dict['facility'], global_features, problem_params)

        # Get value (with problem params)
        value = self.critic(h_dict['individual'], h_dict['facility'], global_features, problem_params)

        # Sample actions
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)

        if deterministic:
            actions = (probs > 0.5).float()
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions).sum()  # Sum over facilities
        entropy = dist.entropy().mean()

        return actions, log_probs, entropy, value

    def get_value(self, obs: Dict[str, Any]) -> Tensor:
        """
        Get state value estimate only.

        Args:
            obs: Observation dictionary

        Returns:
            State value (scalar tensor)
        """
        device = next(self.parameters()).device

        # Convert observation to tensors
        x_dict, edge_index_dict, global_features, problem_params = self._obs_to_tensors(obs, device)

        # Encode graph
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get value (with problem params)
        value = self.critic(h_dict['individual'], h_dict['facility'], global_features, problem_params)

        return value

    def evaluate_actions(
        self,
        obs: Dict[str, Any],
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate given actions for PPO update.

        Args:
            obs: Observation dictionary
            actions: Actions to evaluate [M]

        Returns:
            Tuple of:
                - log_probs: Sum of log probabilities (scalar)
                - entropy: Mean entropy (scalar)
                - value: State value estimate (scalar)
        """
        device = next(self.parameters()).device

        # Convert observation to tensors
        x_dict, edge_index_dict, global_features, problem_params = self._obs_to_tensors(obs, device)

        # Encode graph
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get actor logits (with problem params)
        logits = self.actor(h_dict['facility'], global_features, problem_params)

        # Get value (with problem params)
        value = self.critic(h_dict['individual'], h_dict['facility'], global_features, problem_params)

        # Evaluate actions
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)

        # Ensure actions are on correct device
        if not isinstance(actions, Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=device)
        else:
            actions = actions.to(device)

        log_probs = dist.log_prob(actions).sum()
        entropy = dist.entropy().mean()

        return log_probs, entropy, value

    def get_action_probs(self, obs: Dict[str, Any]) -> Tensor:
        """
        Get action probabilities without sampling.

        Args:
            obs: Observation dictionary

        Returns:
            Probabilities tensor [M]
        """
        device = next(self.parameters()).device

        # Convert observation to tensors
        x_dict, edge_index_dict, global_features, problem_params = self._obs_to_tensors(obs, device)

        # Encode graph
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get actor logits (with problem params)
        logits = self.actor(h_dict['facility'], global_features, problem_params)

        return torch.sigmoid(logits)

    def _obs_to_heterodata(
        self,
        obs: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[HeteroData, Tensor, Tensor]:
        """
        Convert observation dictionary to HeteroData object for batching.

        Args:
            obs: Observation dictionary from environment
            device: Target device

        Returns:
            Tuple of (HeteroData, global_features, problem_params)
        """
        N = obs['num_individuals']
        M = obs['num_facilities']

        # Create HeteroData object
        data = HeteroData()

        # Node features
        ind_feat = obs['individual_features']
        fac_feat = obs['facility_features']

        if isinstance(ind_feat, Tensor):
            data['individual'].x = ind_feat[:N].to(device=device, dtype=torch.float32)
        else:
            data['individual'].x = torch.tensor(ind_feat[:N], dtype=torch.float32, device=device)

        if isinstance(fac_feat, Tensor):
            data['facility'].x = fac_feat[:M].to(device=device, dtype=torch.float32)
        else:
            data['facility'].x = torch.tensor(fac_feat[:M], dtype=torch.float32, device=device)

        # Edge indices
        edge_ii = obs['edge_index_ii']
        edge_if = obs['edge_index_if']
        edge_ff = obs['edge_index_ff']

        if isinstance(edge_ii, Tensor):
            data['individual', 'interacts', 'individual'].edge_index = edge_ii.to(device=device, dtype=torch.long)
        else:
            data['individual', 'interacts', 'individual'].edge_index = torch.tensor(edge_ii, dtype=torch.long, device=device)

        if isinstance(edge_if, Tensor):
            data['individual', 'visits', 'facility'].edge_index = edge_if.to(device=device, dtype=torch.long)
            data['facility', 'visited_by', 'individual'].edge_index = edge_if[[1, 0]].to(device=device, dtype=torch.long)
        else:
            data['individual', 'visits', 'facility'].edge_index = torch.tensor(edge_if, dtype=torch.long, device=device)
            data['facility', 'visited_by', 'individual'].edge_index = torch.tensor(edge_if[[1, 0]], dtype=torch.long, device=device)

        if isinstance(edge_ff, Tensor):
            data['facility', 'connects', 'facility'].edge_index = edge_ff.to(device=device, dtype=torch.long)
        else:
            data['facility', 'connects', 'facility'].edge_index = torch.tensor(edge_ff, dtype=torch.long, device=device)

        # Global features
        glob_feat = obs['global_features']
        if isinstance(glob_feat, Tensor):
            global_features = glob_feat.to(device=device, dtype=torch.float32)
        else:
            global_features = torch.tensor(glob_feat, dtype=torch.float32, device=device)

        # Problem parameters
        problem_params = None
        if 'problem_params' in obs:
            prob_params = obs['problem_params']
            if isinstance(prob_params, Tensor):
                problem_params = prob_params.to(device=device, dtype=torch.float32)
            else:
                problem_params = torch.tensor(prob_params, dtype=torch.float32, device=device)

        return data, global_features, problem_params

    def _batch_observations(
        self,
        observations: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[Batch, Tensor, Tensor, List[int], List[int]]:
        """
        Batch multiple observations into a single Batch object.

        Args:
            observations: List of observation dictionaries
            device: Target device

        Returns:
            Tuple of:
                - batch: Batched HeteroData
                - global_features: [B, D_g] stacked global features
                - problem_params: [B, D_p] stacked problem params
                - num_individuals: List of N for each graph
                - num_facilities: List of M for each graph
        """
        data_list = []
        global_features_list = []
        problem_params_list = []
        num_individuals = []
        num_facilities = []

        for obs in observations:
            data, glob_feat, prob_params = self._obs_to_heterodata(obs, device)
            data_list.append(data)
            global_features_list.append(glob_feat)
            if prob_params is not None:
                problem_params_list.append(prob_params)
            num_individuals.append(obs['num_individuals'])
            num_facilities.append(obs['num_facilities'])

        # Batch the graphs
        batch = Batch.from_data_list(data_list)

        # Stack global features and problem params
        global_features = torch.stack(global_features_list, dim=0)  # [B, D_g]
        if problem_params_list:
            problem_params = torch.stack(problem_params_list, dim=0)  # [B, D_p]
        else:
            problem_params = None

        return batch, global_features, problem_params, num_individuals, num_facilities

    def evaluate_actions_batched(
        self,
        observations: List[Dict[str, Any]],
        actions: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions for a batch of observations in parallel.

        This is much more efficient than processing observations one by one
        because it batches all GNN operations.

        Args:
            observations: List of observation dictionaries
            actions: List of action tensors (one per observation)

        Returns:
            Tuple of:
                - log_probs: [B] log probabilities (sum over facilities for each)
                - entropies: [B] mean entropies for each observation
                - values: [B] state value estimates
        """
        device = next(self.parameters()).device
        batch_size = len(observations)

        # Batch observations
        batch, global_features, problem_params, num_individuals, num_facilities = \
            self._batch_observations(observations, device)

        # Extract batched tensors
        x_dict = {
            'individual': batch['individual'].x,
            'facility': batch['facility'].x,
        }
        edge_index_dict = {
            ('individual', 'interacts', 'individual'): batch['individual', 'interacts', 'individual'].edge_index,
            ('individual', 'visits', 'facility'): batch['individual', 'visits', 'facility'].edge_index,
            ('facility', 'visited_by', 'individual'): batch['facility', 'visited_by', 'individual'].edge_index,
            ('facility', 'connects', 'facility'): batch['facility', 'connects', 'facility'].edge_index,
        }

        # Encode all graphs in parallel (single GNN forward pass!)
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get batch indices for nodes
        batch_individual = batch['individual'].batch  # [total_N] - which graph each individual belongs to
        batch_facility = batch['facility'].batch      # [total_M] - which graph each facility belongs to

        # === ACTOR: Compute per-facility logits ===
        # Expand global features and problem params to match facilities
        # global_features: [B, D_g] -> [total_M, D_g]
        global_features_expanded = global_features[batch_facility]  # [total_M, D_g]

        if problem_params is not None:
            problem_params_expanded = problem_params[batch_facility]  # [total_M, D_p]
        else:
            problem_params_expanded = torch.zeros(
                h_dict['facility'].size(0), self.problem_param_dim, device=device
            )

        # Compute logits for all facilities at once
        combined_actor = torch.cat([
            h_dict['facility'],
            global_features_expanded,
            problem_params_expanded
        ], dim=-1)
        all_logits = self.actor.mlp(combined_actor).squeeze(-1)  # [total_M]

        # === CRITIC: Compute per-graph values ===
        # Pool individual embeddings per graph
        individual_pool = scatter_mean(
            h_dict['individual'], batch_individual, dim=0
        )  # [B, H]

        # Pool facility embeddings per graph
        facility_pool = scatter_mean(
            h_dict['facility'], batch_facility, dim=0
        )  # [B, H]

        # Compute values for all graphs
        if problem_params is not None:
            combined_critic = torch.cat([
                individual_pool,
                facility_pool,
                global_features,
                problem_params
            ], dim=-1)
        else:
            zeros = torch.zeros(batch_size, self.problem_param_dim, device=device)
            combined_critic = torch.cat([
                individual_pool,
                facility_pool,
                global_features,
                zeros
            ], dim=-1)

        values = self.critic.mlp(combined_critic).squeeze(-1)  # [B]

        # === Evaluate actions ===
        # Concatenate all actions
        all_actions = torch.cat(actions, dim=0).to(device)  # [total_M]

        # Compute log probs and entropy
        probs = torch.sigmoid(all_logits)
        dist = Bernoulli(probs=probs)

        all_log_probs = dist.log_prob(all_actions)  # [total_M]
        all_entropies = dist.entropy()  # [total_M]

        # Sum log_probs per graph, mean entropy per graph
        log_probs = scatter_add(all_log_probs, batch_facility, dim=0)  # [B]
        entropies = scatter_mean(all_entropies, batch_facility, dim=0)  # [B]

        return log_probs, entropies, values

    def forward_batched(
        self,
        observations: List[Dict[str, Any]],
        deterministic: bool = False,
    ) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        """
        Batched forward pass for parallel action selection.

        Processes multiple observations in a single GNN forward pass,
        then samples actions for each. Much more efficient than
        sequential calls to forward().

        Args:
            observations: List of observation dictionaries
            deterministic: If True, return argmax actions

        Returns:
            Tuple of:
                - actions: List of binary action tensors (one per observation)
                - log_probs: [B] sum of log probabilities per observation
                - entropies: [B] mean entropies per observation
                - values: [B] state value estimates
        """
        device = next(self.parameters()).device
        batch_size = len(observations)

        # Batch observations
        batch, global_features, problem_params, num_individuals, num_facilities = \
            self._batch_observations(observations, device)

        # Extract batched tensors
        x_dict = {
            'individual': batch['individual'].x,
            'facility': batch['facility'].x,
        }
        edge_index_dict = {
            ('individual', 'interacts', 'individual'): batch['individual', 'interacts', 'individual'].edge_index,
            ('individual', 'visits', 'facility'): batch['individual', 'visits', 'facility'].edge_index,
            ('facility', 'visited_by', 'individual'): batch['facility', 'visited_by', 'individual'].edge_index,
            ('facility', 'connects', 'facility'): batch['facility', 'connects', 'facility'].edge_index,
        }

        # Encode all graphs in parallel (single GNN forward pass!)
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get batch indices for nodes
        batch_individual = batch['individual'].batch  # [total_N]
        batch_facility = batch['facility'].batch      # [total_M]

        # === ACTOR: Compute per-facility logits ===
        global_features_expanded = global_features[batch_facility]  # [total_M, D_g]

        if problem_params is not None:
            problem_params_expanded = problem_params[batch_facility]  # [total_M, D_p]
        else:
            problem_params_expanded = torch.zeros(
                h_dict['facility'].size(0), self.problem_param_dim, device=device
            )

        combined_actor = torch.cat([
            h_dict['facility'],
            global_features_expanded,
            problem_params_expanded
        ], dim=-1)
        all_logits = self.actor.mlp(combined_actor).squeeze(-1)  # [total_M]

        # === CRITIC: Compute per-graph values ===
        individual_pool = scatter_mean(
            h_dict['individual'], batch_individual, dim=0
        )  # [B, H]

        facility_pool = scatter_mean(
            h_dict['facility'], batch_facility, dim=0
        )  # [B, H]

        if problem_params is not None:
            combined_critic = torch.cat([
                individual_pool,
                facility_pool,
                global_features,
                problem_params
            ], dim=-1)
        else:
            zeros = torch.zeros(batch_size, self.problem_param_dim, device=device)
            combined_critic = torch.cat([
                individual_pool,
                facility_pool,
                global_features,
                zeros
            ], dim=-1)

        values = self.critic.mlp(combined_critic).squeeze(-1)  # [B]

        # === Sample actions ===
        probs = torch.sigmoid(all_logits)
        dist = Bernoulli(probs=probs)

        if deterministic:
            all_actions = (probs > 0.5).float()
        else:
            all_actions = dist.sample()

        all_log_probs = dist.log_prob(all_actions)  # [total_M]
        all_entropies = dist.entropy()  # [total_M]

        # Sum log_probs per graph, mean entropy per graph
        log_probs = scatter_add(all_log_probs, batch_facility, dim=0)  # [B]
        entropies = scatter_mean(all_entropies, batch_facility, dim=0)  # [B]

        # Split actions by graph
        actions = []
        offset = 0
        for M in num_facilities:
            actions.append(all_actions[offset:offset + M])
            offset += M

        return actions, log_probs, entropies, values

    def get_value_batched(
        self,
        observations: List[Dict[str, Any]],
    ) -> Tensor:
        """
        Batched value estimation for multiple observations.

        Args:
            observations: List of observation dictionaries

        Returns:
            values: [B] state value estimates
        """
        device = next(self.parameters()).device
        batch_size = len(observations)

        # Batch observations
        batch, global_features, problem_params, num_individuals, num_facilities = \
            self._batch_observations(observations, device)

        # Extract batched tensors
        x_dict = {
            'individual': batch['individual'].x,
            'facility': batch['facility'].x,
        }
        edge_index_dict = {
            ('individual', 'interacts', 'individual'): batch['individual', 'interacts', 'individual'].edge_index,
            ('individual', 'visits', 'facility'): batch['individual', 'visits', 'facility'].edge_index,
            ('facility', 'visited_by', 'individual'): batch['facility', 'visited_by', 'individual'].edge_index,
            ('facility', 'connects', 'facility'): batch['facility', 'connects', 'facility'].edge_index,
        }

        # Encode all graphs in parallel
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Get batch indices
        batch_individual = batch['individual'].batch
        batch_facility = batch['facility'].batch

        # Pool embeddings per graph
        individual_pool = scatter_mean(h_dict['individual'], batch_individual, dim=0)
        facility_pool = scatter_mean(h_dict['facility'], batch_facility, dim=0)

        # Compute values
        if problem_params is not None:
            combined = torch.cat([individual_pool, facility_pool, global_features, problem_params], dim=-1)
        else:
            zeros = torch.zeros(batch_size, self.problem_param_dim, device=device)
            combined = torch.cat([individual_pool, facility_pool, global_features, zeros], dim=-1)

        values = self.critic.mlp(combined).squeeze(-1)  # [B]

        return values
