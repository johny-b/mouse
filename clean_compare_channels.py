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
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")


# %%

policy, hook = load_model()

# %%
venvs, labels = get_venvs('cheese')

# %%

example_ix = 0
print(labels[example_ix])
visualization.visualize_venv(venvs[example_ix][0], render_padding=False)
visualization.visualize_venv(venvs[example_ix][1], render_padding=False)

venv_cnt = len(venvs)

#   NOTE: this list is pretty arbitrary - don't depend on it
INTERESTING_CHANNELS = [121, 21, 35, 73, 17, 7, 112, 80, 71, 123]

# %%
def get_single_act(venv):
    with t.no_grad():
        hook.run_with_input(venv.reset().astype('float32'))
    
    return hook.values_by_label['embedder.relu3_out'][0]

def get_raw_data(venvs):
    data = []
    for venv_1, venv_2 in venvs:
        act_1 = get_single_act(venv_1)
        act_2 = get_single_act(venv_2)
        data.append(t.stack([act_1, act_2]))
    return t.stack(data)    
    
raw_data = get_raw_data(venvs)
assert raw_data.shape == (venv_cnt, 2, 128, 8, 8)
                    
# %%
channel_sums = raw_data.sum((-1,-2))
assert channel_sums.shape == (venv_cnt, 2, 128)

ix = 0
print(channel_sums[ix][0][121])
print(channel_sums[ix][1][121])

vf_1 = visualization.vector_field(venvs[ix][0], policy)
visualization.plot_vf(vf_1)
plt.show()
vf_2 = visualization.vector_field(venvs[ix][1], policy)
visualization.plot_vf(vf_2)

# %%
channel_diff = (channel_sums[:, 0] - channel_sums[:, 1]).abs()
assert channel_diff.shape == (venv_cnt, 128)

# %%
channel = 123
for venv_data, label in zip(channel_sums, labels):
    print(channel, label, venv_data[:,channel].round().to(t.int).tolist())

# %%
#   TODO: fix plotting here
def plot_channel(channel):
    c = channel.unsqueeze(2)
    
    from matplotlib import pyplot as plt
    
    # plt.figure(figsize=(3.8,3.8))
    # x = ax.imshow(c, vmin=0, vmax=8, alpha=0.00)
    # plt.colorbar(x)
    # plt.axis('off')
    # plt.show()
    
    plt.figure(figsize=(3.8, 3.8))
    x = plt.imshow(c, vmin=0, vmax=8)
    plt.colorbar(x)
    plt.show()


# %%
for example_ix in range(16):
    for version_ix in range(2):
        venv = venvs[example_ix][version_ix]
        img, ax = visualization.visualize_venv(venv)
        
        plot_channel(ax, raw_data[example_ix][version_ix][21])
        break
    break
# %%

seed = tools.get_seed_with_decision_square(25)
# seed = 13890400
print(seed)
venv = maze.create_venv(1, seed, 1)
state = maze.state_from_venv(venv)
tools.put_mouse_on_decision_square(state)
venv = maze.venv_from_grid(state.inner_grid())
visualization.visualize_venv(venv)

act = get_single_act(venv)
print(act[121].sum())
plot_channel(act[121])
# %%

venv = maze.create_venv(1, seed, 1)
vf_1 = visualization.vector_field(venv, policy)
visualization.plot_vf(vf_1)

# %%
def get_maze(size, in_distribution):
    while True:
        seed = tools.get_seed_with_decision_square(size)
        venv = maze.create_venv(1, seed, 1)
        
        state = maze.state_from_venv(venv)
        grid = state.inner_grid()
        grid[grid == 2] = 100
        
        corridors = np.where(grid == 100)
        
        if in_distribution:
            available_cheese_positions = list((x, y) for x, y in zip(*corridors) if x >= 20 and y >=20)
        else:
            available_cheese_positions = list((x, y) for x, y in zip(*corridors) if x < 20 or y < 20)
        
        cheese_position_x = np.random.randint(len(available_cheese_positions))
        cheese_position = available_cheese_positions[cheese_position_x]
        grid[cheese_position] = 2
        
        try:
            venv = maze.venv_from_grid(grid)
        except AssertionError:
            continue

        state = maze.state_from_venv(venv)
        try:
            tools.put_mouse_on_decision_square(state)
        except TypeError:
            continue
        
        venv = maze.venv_from_grid(state.inner_grid())
        return seed, venv

def get_maze_data(size, cnt, in_distribution):
    data = []
    while len(data) < cnt:
        seed, venv = get_maze(size, in_distribution)
        
        with t.no_grad():
            categorical, _ = hook.run_with_input(venv.reset().astype('float32'))

        act121 = hook.values_by_label['embedder.relu3_out'][0][121]
        act121_sum = act121.sum().item()
        
        if act121_sum < 30:
            continue
        
        step_to_cheese = tools.next_step_to_cheese(maze.state_from_venv(venv).inner_grid())
        mouse_step = models.human_readable_action(categorical.logits.argmax())
        
        data.append((venv, step_to_cheese, mouse_step))
        # visualization.visualize_venv(venv)
        # print(seed, act121_sum, mouse_step, step_to_cheese)
        print(in_distribution, len(data))
    return data

# %%
MAZE_CNT = 500

in_distr_mazes = get_maze_data(25, MAZE_CNT, True)
with open(f"in_distr_mazes_{MAZE_CNT}.pickle", 'wb') as handle:
    data = [[maze.state_from_venv(row[0]).inner_grid(), row[1], row[2]] for row in in_distr_mazes]
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# visualization.visualize_venv(in_distr_mazes[2][0])

# %%
oo_distr_mazes = get_maze_data(25, MAZE_CNT, False)
with open(f"oo_distr_mazes_{MAZE_CNT}.pickle", 'wb') as handle:
    data = [[maze.state_from_venv(row[0]).inner_grid(), row[1], row[2]] for row in oo_distr_mazes]
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
# visualization.visualize_venv(oo_distr_mazes[2][0])

# %%
from pprint import pprint
pprint([x[1:] for x in in_distr_mazes])

# %%
from collections import Counter
c = Counter([x[1:] for x in oo_distr_mazes])
print(c)
# %%

vals = []
for row in in_distr_mazes:
    if row[1] != 'UP':
        vals.append(row[0])
        
print(len(vals))
# %%
venv = vals[1]
visualization.visualize_venv(venv)

# with open(f"in_distribution_failure.pickle", 'wb') as handle:
#     grid = maze.state_from_venv(venv).inner_grid()
#     pickle.dump(grid, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
vf_1 = visualization.vector_field(venv, policy)
visualization.plot_vf(vf_1)
# %%
from procgen_tools.rollout_utils import rollout_video_clip, get_predict

predict = get_predict(policy)
x = rollout_video_clip(predict, 7, venv=venv)

x[1].write_videofile('my_video.mp4')
# %%
