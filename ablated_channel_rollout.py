# %%
import numpy as np
import pickle
import torch as t
from matplotlib import pyplot as plt

from procgen_tools.imports import load_model
from procgen_tools import visualization, maze, models

from custom_channels import get_venvs

import tools

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%

policy, hook = load_model()

# %%
MAZE_SIZE = 15
CHANNELS = [121]
NUM_SAMPLES = 1000

# %%
def zero_channels(x):
    x[:, CHANNELS] = 0

modified_policy = tools.PolicyWithRelu3Mod(policy, zero_channels)

all_success_cnt = 0
all_fail_cnt = 0
orig_success_cnt = 0
modified_success_cnt = 0

for i in range(NUM_SAMPLES):
    seed = tools.get_seed_with_decision_square(MAZE_SIZE)
    grid = maze.state_from_venv(maze.create_venv(1, seed, 1)).inner_grid()
    grid[grid == 2] = 100
    grid[0, MAZE_SIZE - 1] = 2
    venv_1 = maze.venv_from_grid(grid)
    venv_2 = maze.venv_from_grid(grid)
    # visualization.visualize_venv(venv_1)
    orig_success = tools.rollout(policy, venv=venv_1)
    modified_success = tools.rollout(modified_policy, venv=venv_2)

    if orig_success:
        if modified_success:
            all_success_cnt += 1
        else:
            orig_success_cnt += 1
    else:
        if modified_success:
            modified_success_cnt += 1
        else:
            all_fail_cnt += 1
    print(i + 1, all_success_cnt, all_fail_cnt, orig_success_cnt, modified_success_cnt)
# %%
