# CLAUDE.md

## Project Overview

Research codebase for **Dynamic Facility Location via Reinforcement Learning for Epidemic Suppression in Contact Networks**. Implements an SIV epidemic simulation on heterogeneous graphs (individuals + vaccination facilities) using GNNs, with a PPO-based RL agent that learns to open/close facilities to minimize total cost (infection burden + vaccination + operations + transitions).

**Core idea**: Contact network G(V,E) with SIV compartmental dynamics (S->I, S->V, I->V, V->I, V->S). Vaccination rate depends on distance to nearest open facility via exponential decay. RL agent decides facility open/close at coarse decision intervals.

## File Structure

```
RL-facility-location-epidemic/
├── paper/                    # LaTeX paper (Elsevier elsarticle format)
│   ├── main.tex              # Main document
│   ├── nomenclature.tex      # Acronym definitions (acronym package)
│   ├── refs.bib              # Bibliography (~130 references)
│   └── figures/              # TikZ and PDF figures
│
├── code/                     # All Python code (RUN SCRIPTS FROM HERE)
│   ├── params.py             # Global parameters, ranges, device config
│   ├── funcs.py              # EpidemicEnvironment (MessagePassing-based)
│   ├── main.py               # Entry point for simulations
│   ├── rl/                   # RL module (self-contained imports)
│   │   ├── config.py         # PPOConfig, TrainingConfig, EvalConfig
│   │   ├── gym_env.py        # EpidemicGymEnv (Gymnasium wrapper)
│   │   ├── vec_env.py        # VecEnv (parallel rollouts)
│   │   ├── gnn_networks.py   # HeteroGNNEncoder, GNNActorCritic
│   │   ├── rollout_buffer.py # RolloutBuffer with GAE
│   │   └── ppo_trainer.py    # PPOTrainer
│   └── scripts/              # CLI entry points
│       ├── train_ppo.py      # Training with CLI args
│       ├── policies.py       # Policy abstraction (static + learned)
│       └── evaluate_policies.py  # Evaluation with baseline comparison
│
├── data/saved_problems/      # Saved environment instances
└── outputs/                  # checkpoints/, logs/, figures/ (mostly gitignored)
```

## Code Conventions

- **All imports go in `code/params.py`**. Other files use `from params import *`. Exception: `code/rl/` module has its own imports.
- **Run all scripts from `code/` directory** (e.g., `cd code && python scripts/train_ppo.py`).
- **Device handling**: All tensors on GPU if available. Use `device=device` in constructors, `.to(device)` for external tensors, `.cpu()` before numpy/matplotlib.
- Uses LaTeX rendering in matplotlib (`plt.rcParams['text.usetex'] = True`).
- Research code: minimal type hints, heavy boolean masking for vectorized transitions.

## Development Commands

```bash
# Activate venv and cd to code/
.venv\Scripts\activate  # Windows
cd code

# Run simulation
python main.py

# Train PPO (creates timestamped run folders in outputs/)
python scripts/train_ppo.py --total-timesteps 1000000 --seed 42
python scripts/train_ppo.py --total-timesteps 10000 --n-steps 256 --batch-size 32  # quick test

# Resume training
python scripts/train_ppo.py --resume ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/checkpoint_100000.pt --total-timesteps 200000

# Parallel envs (with optional GPU heat control)
python scripts/train_ppo.py --num-envs 4 --total-timesteps 100000
python scripts/train_ppo.py --num-envs 4 --sequential  # prevents GPU overheating

# Evaluate against baselines
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --num-problems 100
python scripts/evaluate_policies.py --checkpoints ../outputs/checkpoints/run_YYYYMMDD_HHMMSS/final.pt --no-baselines  # learned only

# TensorBoard
tensorboard --logdir ./outputs/logs
```

## Paper Conventions

**Abbreviations**: Managed via `acronym` package in `nomenclature.tex`. Never use raw abbreviations in text.
- `\ac{ABBR}` - auto first/subsequent use
- `\acp{ABBR}` - plural
- `\acf{ABBR}` / `\acs{ABBR}` - force full/short form
- Add new: `\acro{NEW}[NEW]{expansion}` in `nomenclature.tex`

**Citations**: `refs.bib` with `natbib` (`\citet{}` for textual, `\citep{}` for parenthetical).

**Paper status**: Sections through Analytical Model Development are complete. RL Methodology, Experiments, and Conclusions are not yet written.

## Key Architecture Notes

- `EpidemicEnvironment` inherits from `torch_geometric.nn.MessagePassing` for GNN-based epidemic dynamics on a heterogeneous graph with 3 edge types (individual-individual, individual-facility, facility-facility).
- `EpidemicGymEnv` wraps the environment with `MultiBinary(M)` action space (set facility open/closed status). Reward = negative total cost per decision step.
- GNN actor-critic: `HeteroGNNEncoder` (shared) -> `ActorHead` (per-facility logits) + `CriticHead` (global value). Both conditioned on global features and normalized problem parameters.
- Domain randomization: each episode samples N, M, and all epidemic/cost parameters from ranges in `env_params_range` (see `params.py`).
- GPU-optimized pipeline: observations stay as GPU tensors throughout, batched PPO updates via PyG `Batch.from_data_list()`.
- Save/load: `env.save()`/`EpidemicEnvironment.load()` for full serialization; `get_state_dict()`/`set_state_dict()` for lightweight mid-episode checkpoints.
