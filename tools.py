# %%
from typing import Callable, Optional

import numpy as np
import torch as t

from procgen_tools import maze, visualization

# %%
def get_seed(maze_size: int) -> int:
    """Returns a random seed that creates maze of a given size."""
    while True:
        seed = np.random.randint(0, 100000000)
        if maze.get_inner_grid_from_seed(seed=seed).shape[0] == maze_size:
            return seed

def get_vf(seed, policy):
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    return visualization.vector_field(venv, policy)        

def get_squares_by_type(maze_size: int) -> int:
    all_squares = [(x, y) for x in range(maze_size) for y in range(maze_size)]
    even_odd_squares = [square for square in all_squares if not (square[0] % 2) and (square[1] % 2)]
    odd_even_squares = [square for square in all_squares if (square[0] % 2) and not (square[1] % 2)]
    even_even_squares = [square for square in all_squares if square not in (even_odd_squares + odd_even_squares)]
    
    return even_odd_squares, odd_even_squares, even_even_squares

def get_activations_sum(hook, seed, layer_name, positions):
    """Returns mean activation absolute value per channel, for given layer_name and mouse positions"""
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    venv_all, (legal_mouse_positions, _) = maze.venv_with_all_mouse_positions(venv)
    with t.no_grad():
        hook.run_with_input(venv_all.reset().astype('float32'))
        
    raw_activations = hook.values_by_label[layer_name]
    assert len(raw_activations.shape) == 4, "This layer has wrong shape"
    activations = raw_activations.abs().sum(dim=-1).sum(dim=-1)

    data = []
    for mouse_pos, activation in zip(legal_mouse_positions, activations):
        if mouse_pos in positions:
            data.append(activation)
            
    data = t.stack(data)
    data = data.mean(dim=0)
    
    return data

class PolicyWithRelu3Mod(t.nn.Module):
    def __init__(self, orig_policy: t.nn.Module, mod_func: Callable[[t.Tensor], Optional[t.Tensor]]):
        super().__init__()
        self.orig_policy = orig_policy
        self.mod_func = mod_func
        
    def forward(self, x):
        hidden = self.hidden(x)
        
        #   NOTE: everything below is just copied from procgen_tools.models.CategoricalPolicy
        from torch.distributions import Categorical
        import torch.nn.functional as F
        
        logits = self.orig_policy.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)                                
        p = Categorical(logits=log_probs)                                       
        v = self.orig_policy.fc_value(hidden).reshape(-1)                                   
        return p, v
        
    def hidden(self, x):
        embedder = self.orig_policy.embedder
        x = embedder.block1(x)
        x = embedder.block2(x)
        x = embedder.block3(x)
        x = embedder.relu3(x)
        
        modified_x = self.mod_func(x)
        #   If nothing was returned, we assume mod was in place
        x = modified_x if modified_x is not None else x
        
        x = embedder.flatten(x)
        x = embedder.fc(x)
        x = embedder.relufc(x)
        return x
    
    def _ablate_relu3(self, x):
        x[:, self.ablated_channels] = 0
        return x

def assert_same_model_wo_ablations(policy):        
    policy_with_ablations = ModelWithRelu3Ablations(policy, [])
    venv = maze.create_venv(num=1, start_level=get_seed(9), num_levels=1)
    obs = t.from_numpy(venv.reset()).to(t.float32)   

    with t.no_grad():
        categorical_0, value_0 = policy(obs)
        categorical_1, value_1 = policy_with_ablations(obs)
    assert t.allclose(categorical_0.logits, categorical_1.logits)
    assert t.allclose(value_0, value_1)
# %%
