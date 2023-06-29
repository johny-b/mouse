# %%
import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization

from tools import maze_str_to_grid, PolicyWithFCMod
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

base_grid = maze_str_to_grid(CUSTOM_MAZE_STR_2)
grid_1 = base_grid.copy()
grid_1[3, 2] = 51
venv_1 = maze.venv_from_grid(grid_1)
visualization.visualize_venv(venv_1, render_padding=False)

grid_2 = base_grid.copy()
grid_2[2, 3] = 51
venv_2 = maze.venv_from_grid(grid_2)
visualization.visualize_venv(venv_2, render_padding=False)

vf_1 = visualization.vector_field(venv_1, policy)
vf_2 = visualization.vector_field(venv_2, policy)

# %%
def get_activations(venvs):
    activations = []
    for venv in venvs:
        with t.no_grad():
            hook.run_with_input(venv.reset().astype('float32'))
        activations.append(hook.values_by_label["embedder.fc_out"][0])
        
    return t.stack(activations)

def get_cheese_dir_channels(venvs):    
    act = get_activations(venvs)
    diff = (act[0] - act[1]).abs()
    mean = (act[0] + act[1]) / 2
    return diff, mean

channels, mean = get_cheese_dir_channels([venv_1, venv_2])
print(channels.topk(100))

# %%
TOP_CHANNELS = 40
def modify_relu3(x):
    cheese_channels = channels.topk(TOP_CHANNELS).indices  
    x[:, cheese_channels] = 0
    
modified_policy = PolicyWithFCMod(policy, modify_relu3)

vf_1_modified = visualization.vector_field(venv_1, modified_policy)
vf_2_modified = visualization.vector_field(venv_2, modified_policy)

visualization.plot_vfs(vf_1, vf_1_modified)
visualization.plot_vfs(vf_2, vf_2_modified)



# %%