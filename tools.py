# %%
from typing import Callable, Optional

import numpy as np
import torch as t

from procgen_tools import maze, visualization, models

# %%
def get_seed(maze_size: int) -> int:
    """Returns a random seed that creates maze of a given size."""
    while True:
        seed = np.random.randint(0, 100000000)
        if maze.get_inner_grid_from_seed(seed=seed).shape[0] == maze_size:
            return seed

def set_cheese(venv, cheese):
    grid = maze.state_from_venv(venv).inner_grid()
    grid[grid == 2] = 100
    grid[cheese] = 2
    return maze.venv_from_grid(grid)

def get_vf(seed, policy, cheese=None):
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    if cheese is not None:
        venv = set_cheese(venv, cheese)
    return visualization.vector_field(venv, policy)        

def get_squares_by_type(maze_size: int) -> int:
    all_squares = [(x, y) for x in range(maze_size) for y in range(maze_size)]
    even_odd_squares = [square for square in all_squares if not (square[0] % 2) and (square[1] % 2)]
    odd_even_squares = [square for square in all_squares if (square[0] % 2) and not (square[1] % 2)]
    even_even_squares = [square for square in all_squares if square not in (even_odd_squares + odd_even_squares)]
    
    return even_odd_squares, odd_even_squares, even_even_squares

def get_activations_sum(hook, seed, layer_name, positions=None, cheese = None):
    """Returns mean activation absolute value per channel, for given layer_name and mouse positions"""
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    
    if cheese is not None:
        venv = set_cheese(venv, cheese)
        
    venv_all, (legal_mouse_positions, _) = maze.venv_with_all_mouse_positions(venv)
    with t.no_grad():
        hook.run_with_input(venv_all.reset().astype('float32'))
        
    raw_activations = hook.values_by_label[layer_name]
    assert len(raw_activations.shape) == 4, "This layer has wrong shape"
    activations = raw_activations.abs().sum(dim=-1).sum(dim=-1)

    data = []
    for mouse_pos, activation in zip(legal_mouse_positions, activations):
        if positions is None or mouse_pos in positions:
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
    

class PolicyWithFCMod(t.nn.Module):
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
        
        x = embedder.flatten(x)
        x = embedder.fc(x)
        
        modified_x = self.mod_func(x)
        #   If nothing was returned, we assume mod was in place
        x = modified_x if modified_x is not None else x
        
        x = embedder.relufc(x)
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

def maze_str_to_grid(maze_str):
    custom_maze_arr = []
    for line in maze_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_arr = []
        for val in line:
            if val == "0":
                line_arr.append(100)
            elif val == "1":
                line_arr.append(51)
            elif val == "M":
                line_arr.append(25)
            elif val == "C":
                line_arr.append(2)
            else:
                raise ValueError(f"Unexpected value {val}")
        custom_maze_arr.append(line_arr)
        
    return np.array(custom_maze_arr)

def get_seed_with_decision_square(size: int) -> int:
    assert size % 2, "size must be an odd number"

    while True:
        seed = np.random.randint(100000000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] != size:
            seed = np.random.randint(100000000)

        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        state_bytes = venv.env.callmethod("get_state")[0]
        
        if maze.maze_has_decision_square(state_bytes):
            return seed
        
def put_mouse_on_decision_square(state: maze.EnvState) -> None:
    decision_square = maze.get_decision_square_from_maze_state(state)
    padding = maze.get_padding(state.inner_grid())
    state.set_mouse_pos(decision_square[0] + padding, decision_square[1] + padding)

def rollout(policy, seed: int = None, num_steps: int = 256, venv=None) -> bool:
    if venv is None:
        assert seed is not None
        venv = maze.create_venv(num = 1, start_level=seed, num_levels=1)
    else:
        assert seed is None
    
    obs = venv.reset()
    for i in range(num_steps):
        obs = t.tensor(obs, dtype=t.float32)
        with t.no_grad():
            categorical, value = policy(obs)
        
        action = categorical.sample().numpy()
        # action = categorical.logits.argmax().unsqueeze(0).numpy()
        obs, rewards, dones, infos = venv.step(action)
        
        if dones[0]:
            return True
    return False
def compare_policies(seed, policy_1, policy_2):
    venv_1 = maze.create_venv(1, seed, 1)
    venv_2 = maze.create_venv(1, seed, 1)
    vf_1 = visualization.vector_field(venv_1, policy_1)
    vf_2 = visualization.vector_field(venv_2, policy_2)
    visualization.plot_vfs(vf_1, vf_2)

def next_step_to_cheese(grid):
    grid = np.array(grid)
    graph = maze.maze_grid_to_graph(grid)
    venv = maze.venv_from_grid(grid)
    mr, mc = maze.state_from_venv(venv).mouse_pos
    padding = maze.get_padding(grid)
    mr_inner, mc_inner = mr - padding, mc - padding                 
    path_to_cheese = maze.get_path_to_cheese(grid, graph, (mr_inner, mc_inner))
    next_step_x, next_step_y = path_to_cheese[1]

    next_step_x, next_step_y = next_step_x + padding, next_step_y + padding
    
    diff = (next_step_x - mr, next_step_y - mc)
    action = next(key for key, val in models.MAZE_ACTION_DELTAS.items() if val == diff)
    
    return action

def get_single_act(hook, venv, layer_name='embedder.relu3_out'):
    with t.no_grad():
        hook.run_with_input(venv.reset().astype('float32'))
    
    return hook.values_by_label[layer_name][0]