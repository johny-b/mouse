# %%
import os
os.environ["CUDA_AVAILABLE_DEVICES"] = ""

import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization

from tools import maze_str_to_grid, PolicyWithRelu3Mod
# %%

policy, hook = load_model()

# %%

CUSTOM_MAZE_STR_2 = """
    000000000
    111111110
    01C001000
    010101110
    0100M0000
    011101110
    010101000
    010001111
    000100000
"""
CUSTOM_MAZE_STR_3 = """
    0000000000000
    1111111111110
    0000000000000
    1111111111110
    0000000000000
    1111111111110
    01C0010000000
    0101011111110
    0100M00000000
    0111011111110
    0101010000000
    0100011111111
    0001000000000
"""
CUSTOM_MAZE_STR_4 = """
    0000000000000
    1111111111110
    01C0010000000
    0101011111110
    0100M00000000
    0111011111110
    0101010000000
    0100011111111
    0001000000000
    1111111111110
    0000000000000
    1111111111110
    0000000000000

    
"""

maze_setups = [
    (CUSTOM_MAZE_STR_2, (3, 2), (2, 3)),
    (CUSTOM_MAZE_STR_3, (7, 2), (6, 3)),
    (CUSTOM_MAZE_STR_4, (3, 2), (2, 3)),
]
ix = 0

maze_str, first_grid_wall, second_grid_wall = maze_setups[ix]

base_grid = maze_str_to_grid(maze_str)
grid_1 = base_grid.copy()
grid_1[first_grid_wall] = 51
venv_1 = maze.venv_from_grid(grid_1)
# visualization.visualize_venv(venv_1, render_padding=False)

grid_2 = base_grid.copy()
grid_2[second_grid_wall] = 51
venv_2 = maze.venv_from_grid(grid_2)
# visualization.visualize_venv(venv_2, render_padding=False)

# %%
vf_1 = visualization.vector_field(venv_1, policy)
vf_2 = visualization.vector_field(venv_2, policy)

# %%
def get_activations(venvs):
    activations = []
    for venv in venvs:
        with t.no_grad():
            hook.run_with_input(venv.reset().astype('float32'))
        activations.append(hook.values_by_label["embedder.relu3_out"][0])
        
    return t.stack(activations)

def get_cheese_dir_channels(venvs):    
    act = get_activations(venvs)
    diff = (act[0] - act[1]).square().sum(dim=(1,2)).sqrt()
    return diff

channels = get_cheese_dir_channels([venv_1, venv_2])

print(channels.topk(10))

# %%
# TEST 1 - channels selected for this particular maze
def get_mod_func(channels):
    def modify_relu3(x):
        x[:, channels] *= 10
    return modify_relu3

TOP_CHANNELS = 16    
mod_func = get_mod_func(channels.topk(TOP_CHANNELS).indices)
modified_policy = PolicyWithRelu3Mod(policy, mod_func)

vf_1_modified = visualization.vector_field(venv_1, modified_policy)
vf_2_modified = visualization.vector_field(venv_2, modified_policy)

visualization.plot_vfs(vf_1, vf_1_modified)
# visualization.plot_vfs(vf_2, vf_2_modified)



# %%
# TEST 2 - use predefined channels
#   Selected from examples 2-4. Expected to works on them.
example_2_4_channels = [6, 7, 14, 17, 21, 30, 35, 43, 53, 71, 73, 80, 96, 101, 110, 112, 121, 123, 124]

mod_func = get_mod_func(example_2_4_channels)
modified_policy = PolicyWithRelu3Mod(policy, mod_func)

vf_1_modified = visualization.vector_field(venv_1, modified_policy)
vf_2_modified = visualization.vector_field(venv_2, modified_policy)

visualization.plot_vfs(vf_1, vf_1_modified)
# visualization.plot_vfs(vf_2, vf_2_modified)

# %%
