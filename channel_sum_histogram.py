# %%
import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from procgen_tools.imports import load_model
from procgen_tools import visualization, maze, models

import tools

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    
# %%
policy, hook = load_model()

# %%
def get_venv_pair(size, dir_):
    dir_ = dir_.upper()
    
    seed = tools.get_seed_with_decision_square(size)
    grid = maze.state_from_venv(maze.create_venv(1, seed, 1)).inner_grid()
    grid[grid == 2] = 100   # remove the original cheese
    grid[grid == 25] = 100  # remove the original mouse

    corridors = list(zip(*np.where(grid == 100)))
    
    #   Set mouse in a random position where action dir_ is legal
    delta = models.MAZE_ACTION_DELTAS[dir_]
    selected_corridors = [x for x in corridors if (x[0] + delta[0], x[1] + delta[1]) in corridors]
    mouse_pos = random.choice(selected_corridors)
    grid[mouse_pos] = 25
    corridors.remove(mouse_pos)
    
    grid_in = grid.copy()
    grid_oo = grid.copy()
    
    #   Set cheese in a random position
    min_in_distr_coord = size - 5
    available_cheese_positions_in =  [square for square in corridors if square[0] >= min_in_distr_coord and square[1] >= min_in_distr_coord]
    available_cheese_positions_oo = [square for square in corridors if square[0] <  min_in_distr_coord and square[1] <  min_in_distr_coord]
    
    cheese_position_in = random.choice(available_cheese_positions_in)
    cheese_position_oo = random.choice(available_cheese_positions_oo)
    
    grid_in[cheese_position_in] = 2
    grid_oo[cheese_position_oo] = 2
    
    cheese_in_dir_in = tools.next_step_to_cheese(grid_in) == dir_
    cheese_in_dir_oo = tools.next_step_to_cheese(grid_oo) == dir_
    
    venv_in = maze.venv_from_grid(grid_in)
    venv_oo = maze.venv_from_grid(grid_oo)
    
    return seed, venv_in, venv_oo, cheese_in_dir_in, cheese_in_dir_oo

# %%
seed, v1, v2, cheese_in_dir_in, cheese_in_dir_oo = get_venv_pair(25, 'right')
visualization.visualize_venv(v1)
visualization.visualize_venv(v2)
print(cheese_in_dir_in, cheese_in_dir_oo)

# %%
# Easy to spot differences, e.g.:
#   121, "up"
#   17,  "down"
#   73,  "right"
CNT = 200
CHANNEL = 73
DIR = 'right'

data = []
for i in tqdm(range(CNT)):        
    seed, venv_in, venv_oo, cheese_in_dir_in, cheese_in_dir_oo = get_venv_pair(25, DIR)
    in_distr_sum = tools.get_single_act(hook, venv_in)[CHANNEL].sum().item()
    oo_distr_sum = tools.get_single_act(hook, venv_oo)[CHANNEL].sum().item()
    
    data.append([True, in_distr_sum, cheese_in_dir_in])
    data.append([False, oo_distr_sum, cheese_in_dir_oo])

# %%
y = [x[1] for x in data]
x = ["In distribution" if x[0] else "Out of distribution" for x in data]
hue = [f"Cheese is {DIR}" if x[2] else f"Cheese is not {DIR}" for x in data]
plot = sns.stripplot(y = y, x=x, hue=hue, dodge=True, alpha=0.2)
plot.set_title(f'Sum of channel {CHANNEL} vs "is the cheese {DIR}"')
plot.set_ylabel(f'Sum of channel {CHANNEL}')
# %%
