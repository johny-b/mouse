# %%
import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization

from tools import get_seed, set_cheese, get_squares_by_type, get_activations_sum, PolicyWithRelu3Mod, get_vf
# %%

policy, hook = load_model()

# %%
MAZE_SIZE = 11
def get_activations(venvs):
    activations = []
    for venv in venvs:
        with t.no_grad():
            hook.run_with_input(venv.reset().astype('float32'))
        activations.append(hook.values_by_label["embedder.relu3_out"][0])
        
    return t.stack(activations)

stds = []
means = []
for i in range(50):
    print(i)
    seed = get_seed(MAZE_SIZE)
    base_venv = maze.create_venv(num=1, start_level=seed, num_levels=1)

    venvs = []
    for cheese_x in range(2, MAZE_SIZE - 1, 2):
        for cheese_y in range(2, MAZE_SIZE - 1, 2):
            venvs.append(set_cheese(base_venv, (cheese_x, cheese_y)))
        
    act = get_activations(venvs)
    means.append(act.mean(dim=0))
    
    act_mean = act.mean(dim=(-1, -2))

    act = act / (act_mean.reshape(len(venvs), 128, 1, 1) + 0.00000000001)
    std = act.std(dim=0, correction=False)
    channel_std = std.mean(dim=(-1, -2))
    stds.append(channel_std)


stds = t.stack(stds)
means = t.stack(means)
means = means.mean(dim=0)

# %%

seed = get_seed(MAZE_SIZE)
print("SEED", seed)
CHEESE = (2, 8)
TOP_CHANNELS = 32


def reinforce_cheese(x):
    cheese_channels = stds.sum(dim=0).topk(TOP_CHANNELS).indices
    top_right_channels = stds.sum(dim=0).topk(TOP_CHANNELS, largest=False).indices

    old_sum = x.sum()
    x[:, top_right_channels] = 0
    x[:, cheese_channels] *= 4
    x *= (old_sum / x.sum())
    
reinforce_cheese_policy = PolicyWithRelu3Mod(policy, reinforce_cheese)

#   Calculate vector fields and display them
vf_original = get_vf(seed, policy, cheese=CHEESE)
vf_reinforce_cheese = get_vf(seed, reinforce_cheese_policy, cheese=CHEESE)

# visualization.plot_vfs(vf_original, vf_ablate_cheese)
visualization.plot_vfs(vf_original, vf_reinforce_cheese)

# %%
