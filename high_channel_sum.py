# %%
from dataclasses import dataclass
import numpy as np
import random
from typing import Tuple
import pickle
import torch as t
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
def get_maze(size, in_distribution):
    """Return source seed and a venv for a maze with following properties:
        *   Given size
        *   Cheese in random position according to in_distribution:
            *   None -> anywhere
            *   True -> in the top right 5x5
            *   False -> anywhere except top right 5x5
        *   There is a decision square and mouse is placed on it
    """
    while True:
        seed = tools.get_seed_with_decision_square(size)
        venv = maze.create_venv(1, seed, 1)
        
        state = maze.state_from_venv(venv)
        grid = state.inner_grid()
        grid[grid == 2] = 100
        
        corridors = list(zip(*np.where(grid == 100)))
        min_in_distr_coord = size - 5
        if in_distribution is None:
            available_cheese_positions = corridors
        elif in_distribution:
            available_cheese_positions = [square for square in corridors if square[0] >= min_in_distr_coord and square[1] >= min_in_distr_coord]
        else:
            available_cheese_positions = [square for square in corridors if square[0] <  min_in_distr_coord or  square[1] <  min_in_distr_coord]

        cheese_position = random.choice(available_cheese_positions)
        grid[cheese_position] = 2
        
        venv = maze.venv_from_grid(grid)
        
        state = maze.state_from_venv(venv)
        try:
            tools.put_mouse_on_decision_square(state)
        except TypeError:
            #   We got an initial maze with a decision square, but after putting cheese in a random place
            #   it might no longer have a decision square, that's when we get this exception.
            continue
        
        venv = maze.venv_from_grid(state.inner_grid())
        return seed, venv

@dataclass
class MazeData:
    seed: int
    cheese_pos: Tuple[int, int]
    mouse_step: str
    step_to_cheese: str
    step_to_corner: str
    dist: float

def get_maze_data(size, cnt, in_distribution, channel, min_sum=30):
    """Return cnt MazeData with mazes:
        *   Created with `get_maze` function (so with properties described there)
        *   That have the sum of the given channel at least min_sum
            (sum of the channel is calculated using global policy variable)
    """
    data = []
    with tqdm(total=cnt) as pbar:
        while len(data) < cnt:
            seed, venv = get_maze(size, in_distribution)
            
            with t.no_grad():
                categorical, _ = hook.run_with_input(venv.reset().astype('float32'))

            act = hook.values_by_label['embedder.relu3_out'][0][channel]
            act_sum = act.sum().item()
            
            if act_sum < min_sum:
                continue
            
            grid = maze.state_from_venv(venv).inner_grid()
            step_to_cheese = tools.next_step_to_cheese(grid)
            step_to_corner = tools.next_step_to_corner(grid)
            mouse_step = models.human_readable_action(categorical.logits.argmax())
            
            cheese_pos = next(zip(*np.where(grid == 2)))
            top_right = (size - 1, size - 1)
            dist = np.sqrt((cheese_pos[0] - top_right[0]) ** 2 + (cheese_pos[1] - top_right[1]) ** 2)
            
            data.append(MazeData(seed, cheese_pos, mouse_step, step_to_cheese, step_to_corner, dist))

            pbar.update(1)
    return data

# %%
#   Create in distribution data and backup it
MAZE_SIZE = 25
NUM_MAZES = 1000
CHANNEL = 121

in_distr_data = get_maze_data(MAZE_SIZE, NUM_MAZES, True, CHANNEL)
with open(f"in_distr_mazes_{MAZE_SIZE}_{NUM_MAZES}_{CHANNEL}.pickle", 'wb') as handle:
    pickle.dump(in_distr_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
#   Create out of distribution data and backup it
oo_distr_data = get_maze_data(MAZE_SIZE, NUM_MAZES, False, CHANNEL)
with open(f"oo_distr_mazes_{MAZE_SIZE}_{NUM_MAZES}_{CHANNEL}.pickle", 'wb') as handle:
    pickle.dump(in_distr_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# %%
def get_crosstable(data: list[MazeData], dir_):
    cheese_in_dir = [x for x in data if x.mouse_step == x.step_to_cheese == dir_]
    top_right_in_dir = [x for x in data if x.mouse_step == x.step_to_corner == dir_]
    cheese_no_in_dir = [x for x in data if x.mouse_step != dir_ and x.step_to_cheese == dir_]
    top_right_no_in_dir = [x for x in data if x.mouse_step != dir_ and x.step_to_corner == dir_]
    
    return [[len(cheese_in_dir), len(top_right_in_dir)], [len(cheese_no_in_dir), len(top_right_no_in_dir)]]

print(get_crosstable(in_distr_data, 'UP'))
print(get_crosstable(oo_distr_data, 'UP'))

# %%
plot_data = oo_distr_data.copy()

in_distr_oo_distr_ratio = 25 / (MAZE_SIZE ** 2)
plot_data += [x for x in in_distr_data if np.random.random() < in_distr_oo_distr_ratio]

x = [x.dist for x in plot_data]
y = ["Mouse goes to the cheese" if x.mouse_step == x.step_to_cheese else "Mouse goes to the top right" for x in plot_data]
plot = sns.stripplot(x=x, y=y)
plot.set_title(f"Mouse action at the decision point, sum(channel_{CHANNEL}) > 30")
plot.set_xlabel("Euclidean distance between the cheese and the top right corner")
# %%
