# %%
import numpy as np
import torch as t

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization, models

from tools import maze_str_to_grid, PolicyWithRelu3Mod, get_seed_with_decision_square, rollout, compare_policies
from procgen_tools.rollout_utils import rollout_video_clip, get_predict
from custom_channels import get_venvs
# %%

policy, hook = load_model()

# %%
venvs = get_venvs('cheese')
for i, (venv_1, venv_2) in enumerate(venvs):
    if not i % 4:
        print(i)
        visualization.visualize_venv(venv_1, render_padding=False)
        visualization.visualize_venv(venv_2, render_padding=False)

# %%
def get_activations(venvs):
    activations = []
    for venv in venvs:
        with t.no_grad():
            hook.run_with_input(venv.reset().astype('float32'))
        activations.append(hook.values_by_label["embedder.relu3_out"][0])
        
    return t.stack(activations)

def plot_channel(channel):
    from matplotlib import pyplot as plt
    c = channel.unsqueeze(2)
    plt.imshow(c)
    plt.show()
    

def get_cheese_dir_channels(venvs):    
    act = get_activations(venvs)
    #
    diff = (act[0] - act[1]).square().sum(dim=(1,2)).sqrt()
    print("DIFF", diff[123])
    
    # plot_channel(y)
    # print(x)
    # print(y)
    return diff

for channel in [21]:  # , 21, 35, 73, 17, 7, 112, 80, 71, 123]:
    channel_data_1 = []
    channel_data_2 = []
    for rot in range(4):
        rot_data_1 = []
        rot_data_2 = []
        for m in range(4):
            i = 4 * rot + m
            venv_1, venv_2 = venvs[i]
            act = get_activations([venv_1, venv_2])
            diff = (act[0] - act[1]).square().sum(dim=(1,2)).sqrt()
            rot_data_1.append(act[0][channel].sum().round().item())
            rot_data_2.append(act[1][channel].sum().round().item())
        print(channel, rot, rot_data_1)
        print(channel, rot, rot_data_2)
        channel_data_1.append(round(sum(rot_data_1) / 4))
        channel_data_2.append(round(sum(rot_data_2) / 4))
    # print(channel, channel_data_1)
    # print(channel, channel_data_2)        
    

# vf_1 = visualization.vector_field(venv_1, policy)
# vf_2 = visualization.vector_field(venv_2, policy)

# visualization.visualize_venv(venv_1)
# visualization.visualize_venv(venv_2)
# %%
# TEST 1 - channels selected for this particular maze
def get_mod_func(channels):
    def modify_relu3(x):
        x[:, channels] *= 0
    return modify_relu3

# example_2_4_channels = [6, 7, 14, 17, 21, 30, 35, 43, 53, 71, 73, 80, 96, 101, 110, 112, 121, 123, 124]

c = [121, 35, 73, 112, 71]
# mod_func = get_mod_func(example_2_4_channels)
modified_policy = PolicyWithRelu3Mod(policy, get_mod_func(c))

# %%
# vf_1_modified = visualization.vector_field(venv_1, modified_policy)
# vf_2_modified = visualization.vector_field(venv_2, modified_policy)

# visualization.plot_vfs(vf_1, vf_1_modified)
# visualization.plot_vfs(vf_2, vf_2_modified)

# %%

cnt = 0
for i in range(20):
    seed = 23325461  # get_seed_with_decision_square(19)
    result_orig = rollout(policy, seed)
    result_modified = rollout(modified_policy, seed)
    msg = [i]
    if result_orig != result_modified:
        msg += [seed, result_orig, result_modified]
        if result_modified:
            cnt += 1
            msg.append(":-)")
        else:
            cnt -= 1
    else:
        msg.append(result_orig)
    msg.append(cnt)
    print(*msg)

# %%
venv_1, venv_2 = venvs[1]
vf_1 = visualization.vector_field(venv_1, policy)
vf_2 = visualization.vector_field(venv_1, modified_policy)
visualization.plot_vfs(vf_1, vf_2)
# compare_policies(745677, policy, modified_policy)


# %%
predict = get_predict(modified_policy)
x = rollout_video_clip(predict, 47643765)

x[1].write_videofile('/root/t1.mp4')
# %%
