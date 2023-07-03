# %%
import numpy as np
import pickle
import torch as t
import random
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
def get_venv_pair(size):
    seed = tools.get_seed_with_decision_square(size)
    grid = maze.state_from_venv(maze.create_venv(1, seed, 1)).inner_grid()
    grid[grid == 2] = 100   # remove the original cheese
    grid[grid == 25] = 100  # remove the original mouse

    corridors = list(zip(*np.where(grid == 100)))
    
    #   Set mouse in a random position where action UP is legal
    corridors_with_up = [x for x in corridors if (x[0] + 1, x[1]) in corridors]
    mouse_pos = random.choice(corridors_with_up)
    grid[mouse_pos] = 25
    corridors.remove(mouse_pos)
    
    grid_id = grid.copy()
    grid_ood = grid.copy()
    
    #   Set cheese in a random position
    min_in_distr_coord = size - 5
    available_cheese_positions_id =  [square for square in corridors if square[0] >= min_in_distr_coord and square[1] >= min_in_distr_coord]
    available_cheese_positions_ood = [square for square in corridors if square[0] <  min_in_distr_coord and square[1] <  min_in_distr_coord]
    
    cheese_position_id = random.choice(available_cheese_positions_id)
    cheese_position_ood = random.choice(available_cheese_positions_ood)
    
    grid_id[cheese_position_id] = 2
    grid_ood[cheese_position_ood] = 2
    
    cheese_up_id = tools.next_step_to_cheese(grid_id) == 'UP'
    cheese_up_ood = tools.next_step_to_cheese(grid_ood) == 'UP'
    
    venv_id = maze.venv_from_grid(grid_id)
    venv_ood = maze.venv_from_grid(grid_ood)
    
    return seed, venv_id, venv_ood, cheese_up_id, cheese_up_ood

# %%
data_id = []
data_ood = []
for i in range(1000):
    seed, venv_id, venv_ood, cheese_up_id, cheese_up_ood = get_venv_pair(25)
    
    id_121_sum = tools.get_single_act(hook, venv_id)[121].sum().item()
    ood_121_sum = tools.get_single_act(hook, venv_ood)[121].sum().item()
    
    data_id.append([id_121_sum, cheese_up_id])
    data_ood.append([ood_121_sum, cheese_up_ood])
    
    print(i, cheese_up_id, cheese_up_ood, id_121_sum, ood_121_sum)
# %%
plt.hist([x[0] for x in data_id if     x[1]], alpha=0.5, label='Cheese UP')
plt.hist([x[0] for x in data_id if not x[1]], alpha=0.5, label='Cheese not UP')
plt.legend()
plt.xlim([0, 60])
plt.ylim([0, 250])
plt.title(f"Sum of channel 121 in layer relu3 - in distribution, n={len(data_id)}")
# %%
plt.hist([x[0] for x in data_ood if     x[1]], alpha=0.5, label='Cheese UP')
plt.hist([x[0] for x in data_ood if not x[1]], alpha=0.5, label='Cheese not UP')
plt.legend()
plt.xlim([0, 60])
plt.ylim([0, 250])
plt.title(f"Sum of channel 121 in layer relu3 - out of distribution, n={len(data_ood)}")
# %%
