# %%
import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import visualization, maze

from custom_channels import get_venvs

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")


# %%

policy, hook = load_model()

# %%
venvs, labels = get_venvs('cheese')
print(maze.state_from_venv(venvs[1][0]).inner_grid().shape)

from matplotlib import pyplot as plt
obs = venvs[0][0].reset()[0]
plt.imshow(t.from_numpy(obs).permute(1,2,0))
# %%

example_ix = 4
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

# %%
channel_diff = (channel_sums[:, 0] - channel_sums[:, 1]).abs()
assert channel_diff.shape == (venv_cnt, 128)

# %%
channel = 21
for venv_data, label in zip(channel_sums, labels):
    print(channel, label, venv_data[:,channel].round().to(t.int).tolist())

# %%
#   TODO: fix plotting here
def plot_channel(ax, channel):
    from matplotlib import pyplot as plt
    # plt.figure(figsize=(3.8,3.8))
    
    c = channel.unsqueeze(2)
    x = ax.imshow(c, vmin=0, vmax=8, alpha=0.00)
    plt.colorbar(x)
    plt.axis('off')
    plt.show()

for example_ix in range(16):
    for version_ix in range(2):
        venv = venvs[example_ix][version_ix]
        img, ax = visualization.visualize_venv(venv)
        
        plot_channel(ax, raw_data[example_ix][version_ix][21])
        break
    break
# %%
