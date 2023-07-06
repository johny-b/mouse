# %%
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import visualization, maze, models

from custom_mazes import get_venvs

import tools

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# %%
policy, hook = load_model()

# %%
venvs, labels = get_venvs('cheese')

#   Select only the simplest possible venv.
#   This might be important, because this is the only case where cheese is
#   always in the top right 5x5 - other venvs are probably influenced by OOD effect.
venvs = [x for i, x in enumerate(venvs) if i % 5 == 0]
labels = [x for i, x in enumerate(labels) if i % 5 == 0]

# %%
example_ix = 0
print(labels[example_ix])
visualization.visualize_venv(venvs[example_ix][0], render_padding=False)
visualization.visualize_venv(venvs[example_ix][1], render_padding=False)

venv_cnt = len(venvs)

# %%
def get_raw_data(venvs):
    data = []
    for venv_1, venv_2 in venvs:
        act_1 = tools.get_single_act(hook, venv_1)
        act_2 = tools.get_single_act(hook, venv_2)
        data.append(t.stack([act_1, act_2]))
    return t.stack(data)    
    
raw_data = get_raw_data(venvs)
assert raw_data.shape == (venv_cnt, 2, 128, 8, 8)
                    
# %%
channel_sums = raw_data.sum((-1,-2))
assert channel_sums.shape == (venv_cnt, 2, 128)

channel_diff = (channel_sums[:, 0] - channel_sums[:, 1]).abs()
assert channel_diff.shape == (venv_cnt, 128)

channel_diff_sum = channel_diff.mean(dim=0)

# Channels that have the highest average difference between pairs of environments.
# Note that this is calculated only on a single pair of environments (with rotations),
# so this result is in no way general.
# Channels: [121,  21,  80,  35, 112,  73,  71,   7, 123,  17, 110,  66,  30,  98, 96, 101]
top_channels = channel_diff_sum.topk(16)
print(top_channels)

# %%
# What happens when we set top channels to 0?
# NOTE: this doesn't make that much sense - some of the channels never have
#       values around 0, we should probably set them to some reasonable minimum/mean instead.
def zero_channels(x):
    x[:, top_channels.indices] = 0

modified_policy = tools.PolicyWithRelu3Mod(policy, zero_channels)

# Try this on a venv that was not used for selecting channels
venv = get_venvs("cheese")[0][14][1]

vf_1 = visualization.vector_field(venv, policy)
vf_2 = visualization.vector_field(venv, modified_policy)
visualization.plot_vfs(vf_1, vf_2)