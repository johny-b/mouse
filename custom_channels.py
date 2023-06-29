import numpy as np

from procgen_tools import maze
from tools import maze_str_to_grid


CUSTOM_MAZE_STR_2 = """
    000000000
    111111110
    01C001000
    010101110
    0100M0000
    011101110
    010101000
    010101111
    000000000
"""
CUSTOM_MAZE_STR_3 = """
    0000000000000
    1111111111110
    0000000000000
    1111111111110
    0000000000000
    1111111111110
    01C0010000000
    0101011111110
    0100M00000000
    0111011111110
    0101010000000
    0101011111111
    0000000000000
"""
CUSTOM_MAZE_STR_4 = """
    0000000000000
    1111111111110
    01C0010000000
    0101011111110
    0100M00000000
    0111011111110
    0101010000000
    0101011111111
    0000000000000
    1111111111110
    0000000000000
    1111111111110
    0000000000000    
"""
CUSTOM_MAZE_STR_5 = """
    000000000
    111111110
    01C000010
    010111010
    010101010
    010111010
    010000M00
    011111011
    000000000
"""

CORNER_MAZE = """
    00000001C
    110111111
    01M000000
    010111110
    010100010
    010100010
    010100010
    010111110
    000000000
"""

def get_venvs(name):
    if name == 'cheese':
        no_rot_maze_setups = [
            (CUSTOM_MAZE_STR_2, (3, 2), (2, 3)),
            (CUSTOM_MAZE_STR_3, (7, 2), (6, 3)),
            (CUSTOM_MAZE_STR_4, (3, 2), (2, 3)),
            (CUSTOM_MAZE_STR_5, (3, 2), (2, 3)),
        ]
                
        maze_setups = []
        for i in range(4):
            maze_setups += [row + (i,) for row in no_rot_maze_setups]
    elif name == 'corner':
        maze_setups = [
            (CORNER_MAZE, (7,2), (2,7), 0),
        ]
    else:
        raise ValueError(f"wtf is {name}")

    venvs = []
    for ix in range(len(maze_setups)):
        maze_str, first_grid_wall, second_grid_wall, rot_cnt = maze_setups[ix]

        base_grid = maze_str_to_grid(maze_str)
        grid_1 = base_grid.copy()
        grid_1[first_grid_wall] = 51

        grid_2 = base_grid.copy()
        grid_2[second_grid_wall] = 51

        for i in range(rot_cnt):
            grid_1 = np.rot90(grid_1)
            grid_2 = np.rot90(grid_2)

        venvs.append((maze.venv_from_grid(grid_1), maze.venv_from_grid(grid_2)))
    
    return venvs