import numpy as np
import pandas as pd
import csv
import networkx as nx
from math import log
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib import animation, cm
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'serif'
import seaborn as sns
from scipy.sparse import *
from scipy import *
from scipy.sparse import coo_matrix, bmat
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, mannwhitneyu
import itertools
from typing import List
import os
import pickle
import time
import math
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random
import copy
import glob as glob_module
from collections import deque
import datetime
from concurrent.futures import ProcessPoolExecutor
import concurrent
from multiprocessing import Pool
from tqdm import tqdm
import statistics

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import Data, HeteroData
from torch_geometric.visualization import visualize_graph
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx, degree, remove_self_loops, to_undirected, to_scipy_sparse_matrix
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GAE, InnerProductDecoder, global_mean_pool, global_add_pool, MessagePassing
from torch_geometric.explain import Explainer, GNNExplainer

import gymnasium as gym
from stable_baselines3 import DDPG, PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.td3.policies import TD3Policy

from stable_baselines3.common.callbacks import CheckpointCallback

from scipy.sparse.linalg import eigsh

# Markov model parameters
env_params = {
    "N": 70,
    "M": 10,
    "beta_1": 2,
    "beta_2": 0.2,
    "delta": 0.5,
    "omega": 0.05,
    "v_min": 0.1,
    "v_max": 2.0,
    "alpha": 1,
    "f_plus": 2.0,
    "f_minus": 1.0,
    "C_O": 1.0,
    "C_I": 0.1,
    "C_V": 0.003,
    "network_type": "renyi",
    "avg_deg": 7,
    "pct_open_fac": 0.7,
    "dt": 0.01
}

# Ranges for randomization (min, max) tuples
# These ranges are designed to cover scenarios where:
# - Opening facilities is optimal (high C_I, low facility costs)
# - Closing facilities is optimal (low C_I, high facility costs)
# - Mixed strategies are optimal (balanced costs)
env_params_range = {
    # Population/Network
    "N": (1000, 10000),           # Reduced max for faster training
    "M": (5, 30),                 # Moderate facility count
    "avg_deg": (3.0, 20.0),       # Network connectivity
    "network_type": ["expo", "renyi"],

    # Epidemic dynamics - include slow epidemics where vaccination matters
    "beta_1": (0.1, 4.0),         # Infection rate (include slow!)
    "beta_2": (0.05, 0.1),        # Breakthrough infection rate
    "delta": (0.1, 4.0),          # Recovery rate
    "omega": (0.01, 1.0),         # Waning immunity rate

    # Vaccination - v_min near zero forces facility dependence
    "v_min": (0.0001, 0.005),     # Near zero! No facilities = no vaccination
    "v_max": (0.1, 20.0),         # High max for fast vaccination when open
    "alpha": (0.1, 4.0),          # Distance decay parameter

    # Costs - wide ranges to cover all scenarios
    "C_I": (0.1, 100),          # Infection costs
    "C_V": (0.001, 0.05),         # Vaccination cost (keep low)
    "C_O": (0.1, 15.0),           # Include LOW operational costs
    "f_plus": (0.5, 40.0),        # Include LOW opening costs
    "f_minus": (0.5, 25.0),       # Include LOW closing costs

    # Initial conditions
    "pct_open_fac": (0.0, 0.5),   # Start with some facilities closed
}

# a dictionary where keys represent the compartments and values are the index of each compartment.
compartments = {
    "S": np.array([1, 0, 0]),    # Susceptible
    "I": np.array([0, 1, 0]),    # Infected
    "V": np.array([0, 0, 1]),    # Vaccinated
}

# colors assigned to each compartment.
compartments_colors = {
    "S": (51/255, 150/255, 1),      # blue
    "I": (1, 51/255, 51/255),       # red
    "V": (51/255, 204/255, 51/255)  # green
}

# compartment abbreviations.
compartments_abbr = {
    "S": "Susceptible",
    "I": "Infected",
    "V": "Vaccinated"
}

# a dictionary where keys represent the compartments and values
# are the percentage of nodes in each compartment in the initial condition.
# The order of the compartments in the dictionary should be preserved everywhere.
init_compartments = {"S": 0.70, "I": 0.20, "V": 0.1}

# setting the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Current version for save/load compatibility
ENVIRONMENT_SAVE_VERSION = "1.0"