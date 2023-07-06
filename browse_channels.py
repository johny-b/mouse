# %%
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
custom_venvs, labels = get_venvs('cheese')

# %%
CHANNEL = 73

example_ix = 0  # (0-4)
for rot in range(4):
    rotated_ix = example_ix + rot * 5
    venv_1, venv_2 = custom_venvs[rotated_ix]
    label = labels[rotated_ix]

    # visualization.visualize_venv(venv_1, render_padding=False)
    # visualization.visualize_venv(venv_2, render_padding=False)

    # visualization.plot_vf(visualization.vector_field(venv_1, policy))
    # plt.show()
    # visualization.plot_vf(visualization.vector_field(venv_2, policy))
    # plt.show()

    venv_1_act = tools.get_single_act(hook, venv_1)
    venv_2_act = tools.get_single_act(hook, venv_2)
    venv_1_act_sum = venv_1_act.sum(dim=(-1, -2))
    venv_2_act_sum = venv_2_act.sum(dim=(-1, -2))

    first_val = venv_1_act_sum[CHANNEL].round().int().item()
    second_val = venv_2_act_sum[CHANNEL].round().int().item()
    print(CHANNEL, label, first_val, second_val)
# %%
# IMPORTANT CHANNELS
#   121 - UP
#    21 - DOWN or LEFT
#    80 - LEFT but maybe also DOWN
#    35 - UP or LEFT
#   112 - LEFT, maybe UP/DOWN, not RIGHT
#    73 - RIGHT
#    71 - UP
#     7 - UP or RIGHT
#   123 - DOWN
#    17 - DOWN
# %%
