# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for **Dynamic Facility Location via Reinforcement Learning for Epidemic Suppression in Contact Networks**. It implements an epidemic simulation environment using Graph Neural Networks (GNNs) on heterogeneous graphs with two node types: individuals and vaccination facilities. The environment models epidemic spread (SIV compartmental model) on contact networks and vaccination dynamics based on proximity to open facilities.

### Mathematical Problem Formulation

**Contact Network**: Undirected graph G(V, E) where:
- V = {1, 2, ..., n} represents individuals (nodes)
- E represents pairwise contacts (edges)
- For individual i, N_i denotes the neighborhood with |N_i| = number of neighbors
- N_i^I represents the number of infected neighbors

**Compartmental States**: Each individual i has state X_i^t ∈ {S, I, V} where:
- **S (Susceptible)**: Can be infected through contact
- **I (Infected)**: Currently infected and infectious
- **V (Vaccinated)**: Includes both vaccinated individuals AND those recovered from infection (both assumed to have comparable immunity)

**Epidemic Transitions** (continuous-time Markov process):
- S → I at rate β₁ N_i^I (infection from contacts, where β₁ is per-contact infection rate)
- I → V at rate δ (recovery with immunity)
- V → I at rate β₂ N_i^I (breakthrough infection, where β₂ < β₁ reflects partial immunity)
- V → S at rate ω (waning immunity)
- S → V at rate ν_i(L) (vaccination, distance-dependent)

**Spatial Vaccination Model**:
Given M candidate facility locations M = {m_1, ..., m_M} where m_j ∈ R² and individual positions z_i ∈ R², the vaccination rate for individual i when facilities L ⊆ M are open is:

```
ν_i(L) = v_min + (v_max - v_min) * exp(-α * d_i(L))
where d_i(L) = min_{m ∈ L} ||z_i - m||
```

Parameters: v_max (max rate at zero distance), v_min (min rate at infinite distance), α > 0 (decay rate)

**RL Problem**: Discretized time horizon t = 1, 2, ..., T where agent decides which facilities to open/close.

**Cost Structure**:
- f⁺: One-time cost to open a facility
- f⁻: One-time cost to close a facility
- C_O: Operational cost per open facility per timestep
- C_V: Cost per vaccination event
- C_I: Disease burden cost per infected individual per timestep

## Architecture

### Core Components

**Main Environment Class**: `EpidemicEnvironment` (in `funcs.py`)
- Inherits from `torch_geometric.nn.MessagePassing` to leverage GNN message-passing for epidemic dynamics
- Models heterogeneous graph with:
  - **Individual nodes**: State [S, I, V] + 2D spatial coordinates (features: [3 + 2])
  - **Facility nodes**: Open/closed status + 2D spatial coordinates (features: [1 + 2])
  - **Edge types**:
    - `individual-interacts-individual`: Contact network (no attributes)
    - `individual-visits-facility`: Bipartite edges with distance attributes
    - `facility-connects-facility`: Fully connected facility network with distance attributes

**Epidemic Dynamics Implementation**:
- **State representation**: Each individual node has one-hot encoding [S, I, V] indicating current compartment
- **Infection via message passing**: GNN aggregates infected neighbors (N_i^I) to compute infection rates β₁ N_i^I and β₂ N_i^I
- **Stochastic transitions**: Discretized continuous-time Markov process using competing exponentials
  - Transition probability = 1 - exp(-total_rate * dt)
  - Destination chosen proportional to individual transition rates
- **Vaccination coupling**: Vaccination rate ν_i(L) recomputed whenever facility status L changes
- **All five transitions implemented**: S→I, S→V, I→V, V→I, V→S

**Network Generation**:
- Two network types supported: 'renyi' (Erdős-Rényi with proximity bias) and 'expo' (exponential degree distribution)
- Edge probabilities based on spatial proximity with exponential decay: exp(-dist / 0.05)
- Networks preserve expected average degree via probability scaling
- Contact network G(V, E) is undirected and static throughout simulation

**Cost Accounting**:
- In `EpidemicEnvironment.step()`: Tracks C_I * (new infections) + C_V * (new vaccinations) + C_O * (open facilities)
- In `EpidemicGymEnv.step()`: Additionally integrates transition costs f⁺ (opening) and f⁻ (closing) when facility status changes
- Total cost = transition_cost + accumulated_epidemic_cost over decision_interval timesteps

### File Structure

```
RL-facility-location-epidemic/
├── CLAUDE.md                 # This documentation file
├── README.md                 # Project overview
├── .gitignore                # Git ignore patterns
│
├── paper/                    # LaTeX paper files (Elsevier format)
│   ├── main.tex              # Main document
│   ├── nomenclature.tex      # Acronym definitions (uses `acronym` package)
│   ├── refs.bib              # Bibliography (~130 references)
│   └── figures/              # Paper figures (TikZ, PDF)
│       ├── graph_markov.tex
│       ├── graph_markov.pdf
│       ├── markov_model.tex
│       └── markov_model.pdf
│
├── code/                     # All Python code (run scripts from here)
│   ├── params.py             # Global parameters, ranges, device configuration
│   ├── funcs.py              # EpidemicEnvironment implementation
│   ├── main.py               # Entry point for simulations
│   │
│   ├── rl/                   # Reinforcement Learning module
│   │   ├── __init__.py       # Module exports
│   │   ├── config.py         # PPOConfig, TrainingConfig, EvalConfig dataclasses
│   │   ├── gym_env.py        # EpidemicGymEnv (Gymnasium wrapper)
│   │   ├── vec_env.py        # VecEnv (vectorized environment for parallel rollouts)
│   │   ├── gnn_networks.py   # HeteroGNNEncoder, GNNActorCritic networks
│   │   ├── rollout_buffer.py # RolloutBuffer with GAE computation
│   │   └── ppo_trainer.py    # PPOTrainer class with training loop
│   │
│   └── scripts/              # Training and evaluation scripts
│       ├── train_ppo.py      # Training entry point with CLI
│       ├── policies.py       # Policy abstraction (static + learned)
│       └── evaluate_policies.py  # Flexible evaluation with baseline comparison
│
├── data/                     # Data storage
│   ├── raw/                  # Raw input data
│   ├── processed/            # Processed data
│   └── saved_problems/       # Saved environment instances
│
├── outputs/                  # Generated outputs
│   ├── figures/              # Generated figures for paper
│   ├── checkpoints/          # Model checkpoints (gitignored)
│   └── logs/                 # Training logs (gitignored)
│
└── .venv/                    # Virtual environment (gitignored)
```

**IMPORTANT**: All Python scripts should be run from the `code/` directory to ensure imports work correctly.

**Core Files:**

- **`params.py`**: All global parameters, ranges, device configuration, and constant definitions
  - `env_params`: Default environment hyperparameters (N, M, β₁, β₂, δ, ω, costs, etc.)
  - `env_params_range`: Ranges for randomization in multi-instance training
  - `compartments`, `compartments_colors`, `compartments_abbr`: SIV state definitions
  - `device`: Auto-selects CUDA if available

- **`funcs.py`**: Complete `EpidemicEnvironment` implementation (~910 lines)
  - `generate_graph()`: Creates heterogeneous PyTorch Geometric HeteroData object
  - `forward()`: Single-step epidemic dynamics (implements stochastic transitions)
  - `step()`: Multi-step simulation with cost tracking and visualization
    - Supports `return_details=True` for RL integration (returns cost breakdown)
  - `_step_fast()`: GPU-optimized simulation for RL training (minimal CPU-GPU sync)
  - `mean_field()`: Analytical mean-field approximation using ODEs (for validation)
  - `visualize_network()`: Network visualization with compartment coloring
  - `compute_vaccination_rates()`: Distance-based vaccination rate computation
  - `save()` / `load()`: Serialize/deserialize environments to disk (see Save/Load System below)
  - `get_state_dict()` / `set_state_dict()`: Lightweight checkpointing for RL episodes
  - `reset()`: Regenerate graph for new RL episode
  - `EpidemicProblemDataset`: PyTorch Dataset class for batched loading
  - `generate_problem_set()`: Utility to create test problem datasets
  - `load_problem_info()` / `validate_saved_environment()`: Problem inspection utilities

- **`main.py`**: Entry point for running simulations or visualizations

**RL Module (`rl/`):**

- **`config.py`**: Configuration dataclasses
  - `PPOConfig`: PPO hyperparameters (learning_rate, clip_range, n_epochs, etc.)
  - `TrainingConfig`: Training settings (total_timesteps, checkpoint_freq, etc.)
  - `EvalConfig`: Evaluation settings (num_episodes, baselines, etc.)

- **`gym_env.py`**: Gymnasium environment wrapper
  - `EpidemicGymEnv`: Wraps `EpidemicEnvironment` with gym.Env interface
  - Handles action/observation spaces, reward computation, episode management
  - Uses `EpidemicEnvironment.step()` with `return_details=True` for GPU-optimized rollouts
  - Integrates transition costs (f⁺, f⁻) into reward
  - **GPU-optimized**: Observations returned as GPU tensors (no CPU conversion)
  - **GPU-optimized**: `step()` accepts both tensor and numpy actions directly

- **`gnn_networks.py`**: GNN-based actor-critic architecture
  - `HeteroGNNEncoder`: Heterogeneous GNN with HeteroConv + SAGEConv layers
  - `ActorHead`: Per-facility logits for open/close decisions
  - `CriticHead`: Global state value estimation via pooling
  - `GNNActorCritic`: Combined network with shared encoder
  - **GPU-optimized**: `_obs_to_tensors()` handles both GPU tensors and numpy arrays

- **`rollout_buffer.py`**: Experience storage for PPO
  - `RolloutBuffer`: Stores trajectories, computes GAE advantages
  - `EpisodeBuffer`: Tracks episode-level statistics for logging
  - `StepMetricsBuffer`: Tracks step-level cost components and compartment fractions for TensorBoard
  - **GPU-optimized**: Actions stored as tensors (no numpy conversion)

- **`ppo_trainer.py`**: PPO training implementation
  - `PPOTrainer`: Complete training loop with rollout collection, PPO updates, checkpointing, logging
  - Supports `tb_log_dir` and `log_file` parameters for log continuity on resume
  - Checkpoints save TensorBoard and JSONL log paths for seamless resume
  - **GPU-optimized**: Passes tensor actions directly to env (no CPU roundtrip)

### Key Dependencies

**PyTorch Ecosystem**:
- `torch` 2.3.0+cu118: Main deep learning framework
- `torch_geometric` 2.5.3: Heterogeneous graph operations, message passing
- `torch_scatter`, `torch_sparse`, `torch_cluster`: Efficient sparse operations

**Reinforcement Learning**:
- `stable_baselines3` 2.4.0: RL algorithms (available but custom PPO implemented)
- `gymnasium` 1.0.0: Environment interface (implemented in `rl/gym_env.py`)

**Scientific Computing**:
- `numpy` 1.26.4, `scipy` 1.15.3: Numerical operations, ODEs for mean-field
- `networkx` 3.3: Graph utilities
- `pandas` 2.3.0, `scikit-learn` 1.7.0: Data processing, ML utilities

**Visualization**:
- `matplotlib` 3.10.3, `seaborn` 0.13.2: Plotting epidemic trajectories and networks

## Development Commands

### Running Simulations

```bash
# Activate virtual environment (from project root)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Change to code directory for running Python scripts
cd code

# Run main simulation
python main.py
```

### Training PPO Agent

Each training run automatically creates timestamped subfolders for checkpoints and logs:
- `outputs/checkpoints/run_YYYYMMDD_HHMMSS/` - All checkpoints for the run
- `outputs/logs/run_YYYYMMDD_HHMMSS/` - All logs (TensorBoard + JSONL) for the run

```bash
# All commands should be run from the code/ directory
cd code

# Basic training (1M timesteps, domain randomization enabled)
# Creates: outputs/checkpoints/run_YYYYMMDD_HHMMSS/ and outputs/logs/run_YYYYMMDD_HHMMSS/
python scripts/train_ppo.py --total-timesteps 1000000 --seed 42

# Quick test training (smaller scale)
python scripts/train_ppo.py --total-timesteps 10000 --n-steps 256 --batch-size 32

# Training with custom run name
python scripts/train_ppo.py --total-timesteps 500000 --run-name my_experiment_v1

# Training with custom hyperparameters
python scripts/train_ppo.py \
    --total-timesteps 500000 \
    --learning-rate 1e-4 \
    --hidden-dim 256 \
    --num-gnn-layers 4 \
    --checkpoint-freq 25000

# Resume training from checkpoint (automatically uses same run folder)
python scripts/train_ppo.py --resume ../outputs/checkpoints/run_20260121_120000/checkpoint_100000.pt --total-timesteps 200000

# Resume with explicit TensorBoard and log file paths (for custom setups)
python scripts/train_ppo.py --resume ../outputs/checkpoints/run_20260121_120000/checkpoint_100000.pt --total-timesteps 200000 \
    --tb-log-dir ../outputs/logs/run_20260121_120000/tensorboard_20260121_120000 \
    --log-file ../outputs/logs/run_20260121_120000/training_20260121_120000.jsonl
```

### Evaluating Trained Models

```bash
# All commands should be run from the code/ directory
cd code

# Evaluate checkpoint against all static baselines on 100 random problems
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final_1000000.pt --num-problems 100

# Evaluate multiple checkpoints from same run
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/checkpoint_50000.pt ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/checkpoint_100000.pt --num-problems 50

# Evaluate on saved problems (for reproducible benchmarks)
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --problem-dir ../data/saved_problems/

# Select specific baselines only
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --baselines all_open all_closed

# Skip all baselines (only evaluate learned policies)
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --no-baselines

# Stochastic evaluation with multiple runs per problem
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --stochastic --runs-per-problem 5

# Save results to CSV and JSON
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --save-csv --save-json

# Keep generated problems for later reuse
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --num-problems 100 --keep-problems
```

### Python REPL for Exploration

```python
# Run from the code/ directory
from funcs import *
from params import *

# Create environment with default parameters
env = EpidemicEnvironment(randomize=False)

# Create environment with randomized parameters (for RL training)
env = EpidemicEnvironment(randomize=True)

# Visualize network
env.visualize_network(legend=True, save=False)

# Run simulation for 3000 timesteps with visualization
cost, history = env.step(num_dt=3000, verbose=True, visualize=True, save_visualization=False)

# Compare with mean-field approximation
env.step(num_dt=3000, fit_mean_field=True, visualize=True)
```

### Using the RL Module Programmatically

```python
from rl import PPOConfig, TrainingConfig, EpidemicGymEnv, GNNActorCritic, PPOTrainer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Gymnasium environment
env = EpidemicGymEnv(
    randomize=True,           # Domain randomization for robust training
    decision_interval=100,    # 100 epidemic timesteps per RL decision
    max_episode_steps=30,     # Max 30 decisions per episode
)

# Create GNN actor-critic policy
policy = GNNActorCritic(
    hidden_dim=128,
    num_gnn_layers=3,
).to(device)

# Create trainer
config = PPOConfig(learning_rate=3e-4, n_steps=2048)
trainer = PPOTrainer(env=env, policy=policy, config=config, device=device)

# Train
results = trainer.train(total_timesteps=100000, verbose=True)

# Evaluate
eval_results = trainer.evaluate(n_episodes=10)
print(f"Mean reward: {eval_results['mean_reward']:.2f}")
```

### Key API Methods

**EpidemicEnvironment initialization**:
- `randomize=False`: Use parameters from `env_params` dict
- `randomize=True`: Sample parameters from `env_params_range` for domain randomization

**Simulation**:
- `step(num_dt, verbose=True, visualize=True, fit_mean_field=False, save_visualization=True, return_details=False)`: Run forward simulation
  - Returns: `(cost, comp_history)` tuple, or `(cost, comp_history, details)` if `return_details=True`
  - `cost`: Total accumulated cost (C_I + C_V + C_O terms)
  - `comp_history`: List of mean compartment distributions at each timestep
  - `details`: Dict with `individual_cost`, `facility_cost`, `total_infections`, `total_vaccinations`, `current_infected` (RL optimization path)

**Mean-field approximation**:
- `mean_field(num_dt, visualize=True)`: Solve ODEs for deterministic approximation
  - Returns: `(S, I, V, t)` arrays of shape `[time_steps, N_individuals]`

## Important Implementation Details

### Device Handling
All tensors are created on `device` (GPU if available). When adding new functionality:
- Always specify `device=device` in tensor constructors
- Use `.to(device)` for tensors from external sources
- Call `.cpu()` before converting to numpy or matplotlib

### Stochastic Transitions
The `forward()` method implements competing exponential processes:
1. Compute total transition rate from each compartment
2. Sample whether transition occurs: `1 - exp(-total_rate * dt)`
3. If transition occurs, sample which destination using relative rates

**Time discretization**:
- Epidemic dynamics use fine-grained timestep `dt` (default 0.01) for accuracy
- RL decision timesteps are coarser: one decision every T epidemic steps
- Current implementation: Facility status L is fixed during `step()` call
- Future RL wrapper will call `step()` repeatedly, updating L between episodes

### Distance-Based Vaccination
Vaccination rates are precomputed via `_compute_min_dists_to_open_facilities()`:
- Caches minimum distance from each individual to nearest open facility
- Must be recomputed when facility open/closed status changes
- Used in `compute_vaccination_rates()` with exponential decay function

### Network Generation Details
- Proximity-based connections use `exp(-dist / 0.05)` similarity
- For 'renyi': Bernoulli sampling with scaled probabilities to preserve expected degree
- For 'expo': Sample node degrees from exponential distribution, then sample neighbors weighted by proximity
- All edges are symmetrized for undirected networks

### Mean-Field Approximation
The `mean_field()` method provides deterministic ODE-based validation:
- Solves system of 3N coupled ODEs (one per compartment per individual)
- Uses `scipy.integrate.odeint` with adjacency matrix representation
- Tracks probability distributions rather than discrete states
- Useful for comparing stochastic simulation against theoretical predictions
- Note: Mean-field is continuous-time while `forward()` discretizes with timestep dt

### Save/Load System

The codebase includes a comprehensive serialization system for saving and loading environments, designed for creating reproducible test problem sets and DataLoader-compatible RL training.

**Version Control**: `ENVIRONMENT_SAVE_VERSION = "1.0"` constant tracks save format for backward compatibility.

**Saving Environments**:
```python
env = EpidemicEnvironment(randomize=True)
env.save('problems/problem_001.pt', metadata={'difficulty': 'hard', 'seed': 42})
```

**Loading Environments**:
```python
env = EpidemicEnvironment.load('problems/problem_001.pt')
print(env._loaded_metadata)  # Access saved metadata
```

**Save Format** (dictionary stored via `torch.save`):
- `version`: Save format version for compatibility
- `saved_at`: ISO timestamp
- `data`: HeteroData graph (moved to CPU)
- `env_params`, `env_params_range`: All parameters as Python dicts
- `compartments`, `compartments_colors`, `compartments_abbr`, `init_compartments`: State definitions
- `randomize`: Configuration flag
- `min_dists_to_open_facilities`: Cached distances (CPU tensor)
- `metadata`: User-provided metadata dict

**Generating Problem Sets**:
```python
from funcs import generate_problem_set

# Generate 1000 randomized problems with seed for reproducibility
filepaths = generate_problem_set(
    output_dir='training_problems/',
    num_problems=1000,
    randomize=True,
    seed=42,
    verbose=True
)
```

**Loading with DataLoader**:
```python
from funcs import EpidemicProblemDataset
from torch.utils.data import DataLoader

dataset = EpidemicProblemDataset('training_problems/', pattern='*.pt')
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

for batch in loader:
    for env in batch:
        # env is a fully restored EpidemicEnvironment
        cost, history = env.step(num_dt=100, verbose=False, visualize=False)
```

**State Checkpointing** (lightweight, for mid-episode saves):
```python
# Save current state
checkpoint = env.get_state_dict()

# Run simulation
env.step(num_dt=500, verbose=False, visualize=False)

# Restore to checkpoint
env.set_state_dict(checkpoint)
```

**Utility Functions**:
- `load_problem_info(filepath)`: Quick metadata inspection without full load
- `validate_saved_environment(filepath)`: Verify file integrity and forward pass

## PPO with GNN-Based Policy (RL Module)

The `rl/` module implements a complete PPO (Proximal Policy Optimization) system with GNN-based actor-critic networks for learning facility control policies.

### Architecture Overview

```
Observation (HeteroData graph + problem_params)
         │
         ▼
┌─────────────────────────────┐
│    HeteroGNNEncoder         │
│  (3 HeteroConv + SAGEConv)  │
│                             │
│  Edge types processed:      │
│  • individual-interacts-individual │
│  • individual-visits-facility      │
│  • facility-visited_by-individual  │
│  • facility-connects-facility      │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│  ActorHead  │ │ CriticHead  │
│  (per-fac)  │ │  (global)   │
│ +glob_feat  │ │ +glob_feat  │
│ +prob_param │ │ +prob_param │
└──────┬──────┘ └──────┬──────┘
       │               │
       ▼               ▼
   logits [M]     value scalar
       │
       ▼
  Bernoulli(σ(logits))
       │
       ▼
  actions [M] ∈ {0,1}^M
```

**Input dimensions:**
- ActorHead: facility_embedding [M, H] + global_features [8] + problem_params [12] → logits [M]
- CriticHead: pooled_individual [H] + pooled_facility [H] + global_features [8] + problem_params [12] → value [1]

### Gymnasium Environment (`EpidemicGymEnv`)

**Action Space**: `MultiBinary(M)` - binary open/close for each facility
- Action `1` = facility open, `0` = facility closed
- Actions directly set facility status (not toggle)

**Observation Space**: Dictionary containing:
- `individual_features`: [N, 5] - one-hot [S, I, V] + coordinates [x, y]
- `facility_features`: [M, 3] - [open_status, x, y]
- `edge_index_ii`: Contact network edges
- `edge_index_if`, `edge_attr_if`: Individual-facility edges with distances
- `edge_index_ff`, `edge_attr_ff`: Facility-facility edges with distances
- `global_features`: [8] - normalized time, S/I/V fractions, open fraction, mean distance, N, M
- `problem_params`: [12] - normalized problem parameters for policy conditioning (see below)
- `num_individuals`: N (current graph size)
- `num_facilities`: M (current number of facilities)

**Problem Parameters Conditioning** (12 features):
The policy and critic receive normalized problem parameters to adapt to different problem instances:
```python
problem_params = [
    β₁/1.0,              # Infection rate
    β₂/1.0,              # Breakthrough infection rate
    δ/1.0,               # Recovery rate
    ω/1.0,               # Waning immunity rate
    v_max/1.0,           # Max vaccination rate
    v_min/0.1,           # Min vaccination rate
    α/100.0,             # Distance decay parameter
    log(1+C_I)/10,       # Infection cost (log-scaled)
    log(1+C_V)/10,       # Vaccination cost (log-scaled)
    log(1+C_O)/10,       # Operational cost (log-scaled)
    log(1+f_plus)/10,    # Opening cost (log-scaled)
    log(1+f_minus)/10,   # Closing cost (log-scaled)
]
```
This enables the GNN policy to generalize across different epidemic dynamics and cost structures during domain randomization.

**Reward**: Negative cost per decision step
```
reward = -(f⁺ × openings + f⁻ × closings +
           decision_interval × (C_O × open_facs + C_I × infections + C_V × vaccinations))
```

**Episode Termination**:
- `terminated=True`: Infected count reaches 0 (epidemic resolved)
- `truncated=True`: Maximum episode steps reached (default 30)

### PPO Hyperparameters (Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `n_steps` | 2048 | Timesteps per rollout before update |
| `batch_size` | 64 | Mini-batch size for PPO updates |
| `n_epochs` | 10 | PPO epochs per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_range` | 0.2 | PPO surrogate objective clip range |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping norm |
| `hidden_dim` | 128 | GNN hidden dimension |
| `num_gnn_layers` | 3 | Number of HeteroConv layers |

### Training Features

- **Domain Randomization**: New random graphs (N, M, parameters) each episode
  - N ranges from `env_params_range['N']` (default: 1000-10000)
  - M ranges from `env_params_range['M']` (default: 5-30)
  - All epidemic/cost parameters sampled from their ranges (see Parameter Ranges below)
  - Policy receives `problem_params` to condition on instance parameters

**Parameter Ranges** (designed to cover all optimal strategies):
```python
env_params_range = {
    # Population/Network
    "N": (1000, 10000),           # Network size
    "M": (5, 30),                 # Number of facilities
    "avg_deg": (3.0, 20.0),       # Network connectivity

    # Epidemic dynamics - includes slow epidemics where vaccination matters
    "beta_1": (0.1, 4.0),         # Infection rate
    "beta_2": (0.05, 2.0),        # Breakthrough infection rate
    "delta": (0.1, 4.0),          # Recovery rate
    "omega": (0.01, 1.0),         # Waning immunity rate

    # Vaccination - v_min near zero forces facility dependence
    "v_min": (0.0001, 0.005),     # Near zero! No facilities = no vaccination
    "v_max": (0.1, 20.0),         # High max for fast vaccination when open
    "alpha": (0.1, 4.0),          # Distance decay parameter

    # Costs - wide ranges to cover all scenarios
    "C_I": (0.1, 100),            # Infection cost (high = open facilities)
    "C_V": (0.001, 0.05),         # Vaccination cost (keep low)
    "C_O": (0.1, 15.0),           # Operational cost
    "f_plus": (0.5, 40.0),        # Opening cost
    "f_minus": (0.5, 25.0),       # Closing cost

    # Initial conditions
    "pct_open_fac": (0.0, 0.5),   # Start with some facilities closed
}
```

These ranges ensure the policy sees problems where:
- **Opening facilities is optimal**: High C_I, low facility costs, slow epidemic
- **Closing facilities is optimal**: Low C_I, high facility costs, fast epidemic
- **Mixed strategies are optimal**: Balanced costs

- **Variable Graph Sizes**: Handles different N and M between episodes
  - Observation dict includes `num_individuals` and `num_facilities`
  - Rollout buffer stores variable-length actions as lists
  - GNN encoder processes arbitrary graph sizes
- **Checkpointing**: Automatic saves every `checkpoint_freq` timesteps
- **Logging**: JSONL format with all metrics for plotting
- **TensorBoard**: Real-time training curves visualization
- **Resume Training**: Load from checkpoint and continue
  - Use `--tb-log-dir` and `--log-file` to maintain same log files
  - Checkpoints store log paths for seamless continuation
- **Early Stopping**: Optional KL divergence threshold

### TensorBoard Live Training Curves

When training, TensorBoard logs are automatically created. To view live training curves:

```bash
# Start training in one terminal (from code/ directory)
cd code
python scripts/train_ppo.py --total-timesteps 100000

# In another terminal, start TensorBoard (from project root)
tensorboard --logdir ./outputs/logs

# Open http://localhost:6006 in your browser

# To compare multiple training runs side-by-side:
tensorboard --logdir_spec "run_v1:outputs/logs/tensorboard_YYYYMMDD_HHMMSS,run_v2:outputs/logs/run_v2/tensorboard_YYYYMMDD_HHMMSS"
```

**Available Metrics in TensorBoard:**
- `losses/policy_loss`, `losses/value_loss`, `losses/total_loss`
- `policy/entropy`, `policy/approx_kl`, `policy/clip_fraction`
- `episode/mean_reward`, `episode/std_reward`, `episode/mean_length`
- `time/fps`, `time/episodes`
- `costs/infection`, `costs/vaccination`, `costs/operation`, `costs/transition` (raw, unscaled)
- `compartments/susceptible`, `compartments/infected`, `compartments/vaccinated` (fractions)
- `facilities/open_fraction`

### Logged Metrics

Training logs (JSONL format) include:
- `policy_loss`: Clipped surrogate objective loss
- `value_loss`: Value function MSE loss
- `entropy`: Policy entropy (exploration measure)
- `approx_kl`: Approximate KL divergence (old vs new policy)
- `clip_fraction`: Fraction of clipped ratios
- `mean_reward`, `std_reward`: Episode statistics
- `mean_length`: Average episode length
- `mean_infection_cost`, `mean_vaccination_cost`, `mean_operation_cost`, `mean_transition_cost`: Raw cost components
- `mean_susceptible_frac`, `mean_infected_frac`, `mean_vaccinated_frac`: Compartment fractions
- `mean_open_facility_frac`: Fraction of facilities open

### Policy Evaluation System

The `scripts/` directory contains a flexible evaluation system with policy abstraction:

**Policy Abstraction (`scripts/policies.py`)**:
```
Policy (abstract base class)
├── StaticPolicy (rule-based strategies)
│   ├── AllOpenPolicy      - Keep all facilities open
│   ├── AllClosedPolicy    - Keep all facilities closed
│   ├── RandomPolicy       - Random decisions each step
│   └── FixedInitialPolicy - Maintain initial configuration
│
└── LearnedPolicy (trained models)
    └── PPOPolicy          - GNNActorCritic from checkpoint
    # Future: DQNPolicy, A2CPolicy, etc.
```

**Adding New Static Strategies**:
```python
# In scripts/policies.py
class MyCustomPolicy(StaticPolicy):
    def __init__(self):
        super().__init__("my_custom")

    def select_action(self, obs, deterministic=True):
        M = obs["num_facilities"]
        # Custom logic here
        return np.array([...], dtype=np.int32)

# Register it
STATIC_POLICIES["my_custom"] = MyCustomPolicy
```

**Adding New RL Methods**:
```python
# In scripts/policies.py
class DQNPolicy(LearnedPolicy):
    @classmethod
    def from_checkpoint(cls, path, device):
        # Load DQN-specific checkpoint format
        ...

# Register it
LEARNED_POLICY_LOADERS["dqn"] = DQNPolicy
```

**Evaluation Script (`scripts/evaluate_policies.py`)**:
- Generates problems and saves to temp directory (or uses `--problem-dir`)
- For each (policy, problem) pair, loads a **fresh copy** of the problem
- Ensures all policies see identical initial states for fair comparison
- Outputs ranking table, detailed metrics, and pairwise comparisons

**Output Example**:
```
================================================================================
                         EVALUATION SUMMARY
================================================================================
Problems: 100 random | Seed: 42 | Decision interval: 100
--------------------------------------------------------------------------------

OVERALL RANKING (by mean total cost, lower is better):
--------------------------------------------------------------------------------
Rank  Policy               Mean Cost    Std Cost   Mean Reward   Win Rate
--------------------------------------------------------------------------------
1     checkpoint_150016      2.45        0.87        -2.45        65.0%
2     all_open               3.12        1.02        -3.12        20.0%
3     all_closed             4.56        1.89        -4.56         8.0%
4     random                 6.23        2.34        -6.23         7.0%
--------------------------------------------------------------------------------
```

### GPU Optimization

The training pipeline is fully GPU-optimized to minimize CPU-GPU data transfers:

**Data Flow (optimized)**:
```
Policy outputs tensor action (GPU)
         │
         ▼ (no conversion)
env.step() accepts tensor directly
         │
         ▼ (stays on GPU)
Observation returned as GPU tensors
         │
         ▼ (no conversion)
Policy processes GPU tensors
         │
         ▼
RolloutBuffer stores GPU tensors
```

**Key optimizations**:
- `_get_observation()` returns GPU tensors (not numpy arrays)
- `env.step()` accepts both `torch.Tensor` and `np.ndarray` actions
- `_obs_to_tensors()` detects input type and skips conversion if already tensor
- Actions stored as tensors in `RolloutBuffer`
- Facility status comparison done on GPU
- `funcs.py`: `generate_graph()` uses `np.array()` before `torch.from_numpy()` to avoid slow list-of-arrays conversion

**Batched PPO Update** (major optimization):
- `evaluate_actions_batched()` processes entire mini-batch in single GNN forward pass
- Uses PyG's `Batch.from_data_list()` to combine variable-size graphs
- `scatter_mean/scatter_add` for per-graph pooling and aggregation
- Eliminates sequential Python loop over observations
- `.item()` calls only after backward pass (no sync during forward)

**Vectorized Environments** (`VecEnv`):
The `rl/vec_env.py` module provides parallel rollout collection:
- `VecEnv` manages multiple `EpidemicGymEnv` instances
- Uses `ThreadPoolExecutor` for parallel environment stepping
- `step_and_reset()` combines step and auto-reset in single parallel call
- `forward_batched()` in GNNActorCritic processes all observations in single GNN pass
- `get_value_batched()` for efficient bootstrap value computation
- **Concurrency control**: `sequential` and `max_concurrent` parameters to prevent GPU overheating

**Usage:**
```bash
# All commands run from code/ directory
cd code

# Train with 4 parallel environments (full parallelism)
python scripts/train_ppo.py --num-envs 4 --total-timesteps 100000

# Train with 4 envs but step sequentially (prevents GPU overheating)
python scripts/train_ppo.py --num-envs 4 --sequential --total-timesteps 100000

# Train with 8 envs, max 2 concurrent (balance between variety and heat)
python scripts/train_ppo.py --num-envs 8 --max-concurrent 2 --total-timesteps 100000
```

```python
from rl import VecEnv, GNNActorCritic, PPOTrainer, PPOConfig

# Full parallelism (original behavior)
vec_env = VecEnv(num_envs=4, randomize=True, decision_interval=100)

# Sequential mode - prevents GPU overheating
vec_env = VecEnv(num_envs=4, randomize=True, sequential=True)

# Limited concurrency - step at most 2 envs at a time
vec_env = VecEnv(num_envs=8, randomize=True, max_concurrent=2)

# PPOTrainer automatically detects VecEnv
trainer = PPOTrainer(env=vec_env, policy=policy, config=config, device=device)
print(f"Using {trainer.num_envs} parallel environments")
```

**Parallelization architecture:**
```
VecEnv.step_and_reset(actions)
         │
         ▼ (ThreadPoolExecutor, limited by max_concurrent)
┌────────┬────────┬────────┐
│ Env 0  │ Env 1  │ Env 2  │ ...  (parallel or sequential)
│ step() │ step() │ step() │
└────────┴────────┴────────┘
         │
         ▼ (gather results)
Batched observations → forward_batched() → Single GNN pass → Batched actions
```

**Concurrency options:**
| Option | Effect | Use case |
|--------|--------|----------|
| `sequential=True` | Step envs one at a time | GPU overheating, debugging |
| `max_concurrent=N` | Limit concurrent steps to N | Balance throughput vs heat |
| Neither (default) | Full parallelism | Maximum throughput |

### Current State

- **Implemented**: Complete PPO training pipeline with GNN policy
- **Validated**: Training loop functional with domain randomization
- **GPU Optimization**: Full pipeline optimized to minimize CPU-GPU transfers
- **Vectorized Environments**: Parallel rollout collection with `VecEnv` and batched GNN inference
- **Problem Parameters**: Policy/critic conditioned on 12 normalized problem parameters
- **Variable Graph Sizes**: Handles different N and M between episodes
- **Transition costs**: f⁺ and f⁻ integrated into reward
- **Domain Randomization**: Fully functional with variable-size environments
- **Resume Continuity**: TensorBoard and JSONL logs continue seamlessly on resume
- **Step Metrics**: Cost components and compartment fractions logged to TensorBoard
- **Training Runs**:
  - `run_v1` (outputs/checkpoints/): Initial training with original parameter ranges (145k+ timesteps)
  - `run_v2` (outputs/checkpoints/run_v2/, outputs/logs/run_v2/): Training with updated parameter ranges designed to cover all optimal strategies
- **Next steps**: Hyperparameter tuning, longer training runs, policy analysis

## Code Style Notes

- **All imports go in `code/params.py`**: When adding new package/module imports, always add them to `params.py`, not in individual files. Other files import from params via `from params import *`
- **Exception: `code/rl/` module**: The RL module is self-contained with its own imports. Import from it via `from rl import ...`
- Uses LaTeX rendering in matplotlib (`plt.rcParams['text.usetex'] = True`)
- Gradient anomaly detection enabled (`torch.autograd.set_detect_anomaly(True)`)
- Heavy use of boolean masking for vectorized state transitions
- Type hints are minimal; code is research-oriented rather than production-ready

## Paper (`paper/`)

The paper is written in LaTeX using the Elsevier `elsarticle` document class.

### Paper Structure and Status

| Section | Label | Status |
|---------|-------|--------|
| Introduction | `sec:intro` | Complete |
| Literature Review | `sec:LitRev` | Complete |
| Problem Statement | `sec:ProbStat` | Complete |
| Analytical Model Development | `sec:AnalyticalModelDev` | Complete |
| Mean-Field Approximation | (subsection) | Complete |
| RL Methodology | `sec:RLMethod` | Not yet written |
| Experiments | `sec:Experiments` | Not yet written |
| Conclusions | `sec:Conclusions` | Not yet written |

### Abbreviation Management

All abbreviations are managed using the `acronym` LaTeX package. Definitions are in `nomenclature.tex`.

**Defined Acronyms:**

| Category | Acronym | Expansion |
|----------|---------|-----------|
| Epidemic Models | SIV | susceptible-infected-vaccinated |
| | SIR | susceptible-infected-recovered |
| | SEIR | susceptible-exposed-infected-recovered |
| | DFE | disease-free equilibrium |
| ML/RL | RL | reinforcement learning |
| | GNN | graph neural network |
| | ANN | artificial neural network |
| | PPO | Proximal Policy Optimization |
| | TD3 | twin delayed deep deterministic policy gradient |
| | DQN | deep Q-network |
| | DRL | deep reinforcement learning |
| Optimization | MILP | mixed-integer linear programming |
| | TSP | traveling salesman problem |
| | VRP | vehicle routing problem |
| | ODE | ordinary differential equation |
| Other | EMS | emergency medical services |

**Usage Rules:**
- **First use**: Use `\ac{ABBR}` which expands to "full form (ABBR)"
- **Subsequent uses**: Use `\ac{ABBR}` which shows only "ABBR"
- **Plural form**: Use `\acp{ABBR}` for "ABBRs" (e.g., `\acp{GNN}` → "GNNs")
- **Force full form**: Use `\acf{ABBR}` to always show full expansion
- **Force short form**: Use `\acs{ABBR}` to always show abbreviation only

**Adding New Abbreviations:**
```latex
% In nomenclature.tex
\acro{NEW}[NEW]{new abbreviation expansion}
```

**IMPORTANT**: Never use raw abbreviations in the paper text. Always use `\ac{}` commands for consistency.

### Bibliography

References are in `refs.bib` using BibTeX format. The paper uses `abbrvnat` bibliography style with `natbib` for author-year citations.

**Citation Commands:**
- `\cite{key}` - Numbered citation: [1]
- `\citet{key}` - Textual: Author (year) or Author [1]
- `\citep{key}` - Parenthetical: (Author, year) or [1]

### Figures

Figures are in `paper/figures/`. The Markov model diagrams are created using TikZ:
- `markov_model.tex` / `.pdf` - SIV state transition diagram
- `graph_markov.tex` / `.pdf` - Contact network schematic

To regenerate PDFs from TikZ source, compile the `.tex` files with `pdflatex`.

## Troubleshooting

### Common Issues

**CUDA out of memory during training**:
- Reduce `hidden_dim` (default 128 → 64)
- Reduce `n_steps` (default 2048 → 1024)
- Use smaller problems via `env_params` with lower N

**GPU overheating with parallel environments**:
- Use `--sequential` flag to step environments one at a time
- Use `--max-concurrent N` to limit concurrent GPU operations (e.g., `--max-concurrent 2`)
- Reduce `--num-envs` to use fewer parallel environments
- Example: `python scripts/train_ppo.py --num-envs 4 --sequential`

**Slow training**:
- Ensure CUDA is available (`torch.cuda.is_available()`)
- Use smaller `decision_interval` for faster episodes
- Reduce `num_gnn_layers` (default 3 → 2)
- Check for GPU-CPU transfer bottlenecks (should be minimal with current optimizations)

**Import errors in rl module**:
- Run from the `code/` directory (not the repository root)
- Ensure you `cd code` before running any Python scripts

**"Creating tensor from list of numpy.ndarrays is slow" warning**:
- This was fixed in `funcs.py` by using `np.array()` before `torch.from_numpy()`
- If you see this warning elsewhere, convert list of arrays to single array first