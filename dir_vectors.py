# %%
from procgen_tools.imports import load_model
from procgen_tools import maze, visualization

from tools import get_seed, get_squares_by_type, get_activations_sum, PolicyWithRelu3Mod, get_vf
# %%

policy, hook = load_model()

# %%
# Find channels that are important for (even, odd) fields and for (odd, even) fields.
MAZE_SIZE = 13
NUM_CHANNELS = 20

# NOTE: embedder.relu3_out is hardcoded inside PolicyWithRelu3Mod
LAYER_NAME = "embedder.relu3_out"  

seed = get_seed(MAZE_SIZE)
print("SEED", seed)
even_odd_squares, odd_even_squares, _ = get_squares_by_type(MAZE_SIZE)

#   Calculate mean sum of activations per channel, on given squares
#   These values are more-or-less "how important is this channel for these squares"
even_odd = get_activations_sum(hook, seed, LAYER_NAME, even_odd_squares)
odd_even = get_activations_sum(hook, seed, LAYER_NAME, odd_even_squares)

#   Compare
diff = even_odd - odd_even
even_odd_channels = diff.topk(NUM_CHANNELS)
odd_even_channels = diff.topk(NUM_CHANNELS, largest=False)

print(f"Channels important for (even, odd) fields: {even_odd_channels.indices.tolist()}")
print(f"Channels important for (odd, even) fields: {odd_even_channels.indices.tolist()}")

# %%
#   ABLATIONS TEST
#   Create modified policies
def ablate_even_odd(x): x[:, even_odd_channels.indices] = 0
def ablate_odd_even(x): x[:, odd_even_channels.indices] = 0

ablate_even_odd_policy = PolicyWithRelu3Mod(policy, ablate_even_odd)
ablate_odd_even_policy = PolicyWithRelu3Mod(policy, ablate_odd_even)

#   Calculate vector fields and display them
vf_original = get_vf(seed, policy)
vf_ablate_even_odd = get_vf(seed, ablate_even_odd_policy)
vf_ablate_odd_even = get_vf(seed, ablate_odd_even_policy)

visualization.plot_vfs(vf_original, vf_ablate_even_odd)
visualization.plot_vfs(vf_original, vf_ablate_odd_even)

# %%
#   REINFORCMENT TEST
def reinforce_even_odd(x): x[:, even_odd_channels.indices] *= 3
def reinforce_odd_even(x): x[:, odd_even_channels.indices] *= 3

reinforce_even_odd_policy = PolicyWithRelu3Mod(policy, reinforce_even_odd)
reinforce_odd_even_policy = PolicyWithRelu3Mod(policy, reinforce_odd_even)

#   Calculate vector fields and display them
vf_original = get_vf(seed, policy)
vf_reinforce_even_odd = get_vf(seed, reinforce_even_odd_policy)
vf_reinforce_odd_even = get_vf(seed, reinforce_odd_even_policy)

visualization.plot_vfs(vf_original, vf_reinforce_even_odd)
visualization.plot_vfs(vf_original, vf_reinforce_odd_even)
# %%
