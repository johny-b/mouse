# %%
import numpy as np
import pickle
import torch as t
from matplotlib import pyplot as plt

from procgen_tools.imports import load_model
from procgen_tools import visualization, maze, models

from custom_mazes import get_venvs

import tools

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %%

policy, hook = load_model()

# %%
custom_venvs, labels = get_venvs('cheese')

#   Simplest maze in the custom mazes dataset, rotation "up vs left"
example_ix = 5
venv_1, venv_2 = custom_venvs[example_ix]

# %%
visualization.visualize_venv(venv_1, render_padding=False)
visualization.visualize_venv(venv_2, render_padding=False)

# %%
visualization.plot_vf(visualization.vector_field(venv_1, policy))
plt.show()
visualization.plot_vf(visualization.vector_field(venv_2, policy))

# %%
venv_1_act = tools.get_single_act(hook, venv_1)
venv_2_act = tools.get_single_act(hook, venv_2)
venv_1_act_sum = venv_1_act.sum(dim=(-1, -2))
venv_2_act_sum = venv_2_act.sum(dim=(-1, -2))

print(venv_1_act_sum[121])
print(venv_2_act_sum[121])

# print((venv_1_act_sum - venv_2_act_sum).abs()[121])

# %%
# tools.plot_channel(venv_1_act[121])
# tools.plot_channel(venv_2_act[121])

# %%
