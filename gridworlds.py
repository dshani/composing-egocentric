"""
@author: Daniel Shani
"""
import numpy as np
import random
from collections import deque


grid_rep_color = np.array(
    [[4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
      5., 5.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
      1., 0., 0., 0., 0., 2.],
     [4., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
      1., 0., 0., 0., 0., 2.],
     [4., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1.,
      1., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
      1., 0., 1., 0., 0., 2.],
     [4., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
      1., 0., 1., 0., 0., 2.],
     [4., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0.,
      1., 1., 1., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
      1., 0., 0., 0., 0., 2.],
     [4., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
      1., 0., 0., 0., 0., 2.],
     [4., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1.,
      1., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 2.],
     [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., -1., 2.],
     [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
      3., 3., 3., 3., 3., 2.]])


grid_rep_color_2 = np.array([[4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                              5., 5., 5., 5.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                              1., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
                              0., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
                              1., 1., 0., 2.],
                             [4., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
                              0., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
                              0., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
                              1., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., -1., 2.],
                             [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
                              3., 3., 3., 2.]])

grid_rep_color_3 = np.array([[4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                              5., 5., 5., 5.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., -1., 2.],
                             [4., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                              1., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
                              0., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
                              1., 1., 0., 2.],
                             [4., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
                              0., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
                              0., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
                              1., 1., 0., 2.],
                             [4., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 2.],
                             [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
                              3., 3., 3., 2.]])

from parameters import parameters


def create_u_barrier(grid, x, y, width = 3, k=0, color=1, size=parameters.size):
    if np.all(grid[max(x - 1, 0):min(x + width + 1, size), max(y - 1, 0):min(y + width + 1, size)] == 0):
        block = np.zeros((width, width))
        block[0, :] = color
        block[-1, :] = color
        block[:, -1] = color
        block = np.rot90(block, k)
        grid[x:x + width, y:y + width] = block
        return grid
    else:
        return None


def generate_random_barriers(grid, N=4, max_steps=1000, color=1):
    grid2 = grid.copy()
    i = 0
    steps = 0
    while i < N and steps < max_steps:
        steps += 1
        x = np.random.randint(1, grid.shape[0] - 3)
        y = np.random.randint(1, grid.shape[1] - 3)
        k = np.random.randint(0, 4)
        new_grid = create_u_barrier(grid2, x, y, k, color)
        if new_grid is not None:
            grid2 = new_grid
            i += 1
        else:
            continue
    return grid2


def generate_random_gridworld(size, N=4, max_steps=500, color=True, reward="br"):
    """

    Args:
        size: length of square gridworld
        N: max number of barriers
        max_steps: number of attempts at getting N barriers
        color: whether diff walls and barriers have diff colors
        reward: "tr", "tl", "br", "bl", "rand_corner" reward location

    Returns:
            gridworld
    """
    grid = np.zeros((size, size))
    if color:
        grid[:, 0] = 2
        grid[:, -1] = 3
        grid[0, :] = 4
        grid[-1, :] = 5
        grid = generate_random_barriers(grid, N=N, max_steps=max_steps, color=1)
    else:
        grid[:, 0] = 1
        grid[:, -1] = 1
        grid[0, :] = 1
        grid[-1, :] = 1
        grid = generate_random_barriers(grid, N=N, max_steps=max_steps, color=1)

    if reward == "rand_corner":
        i = np.random.randint(0, 4)
        reward = ["tl", "tr", "bl", "br"][i]

    if reward == "tl":
        grid[1, 1] = -1
    elif reward == "tr":
        grid[1, -2] = -1
    elif reward == "bl":
        grid[-2, 1] = -1
    elif reward == "br":
        grid[-2, -2] = -1
    return grid.astype(float)

def generate_random_gridworld_(size, N=4, max_steps=500, color=True, reward="br", align=False, width=[3]):
    """

    Args:
        size: length of square gridworld
        N: max number of barriers
        max_steps: number of attempts at getting N barriers
        color: whether diff walls and barriers have diff colors
        reward: "tr", "tl", "br", "bl", "rand_corner" reward location
        align: False, True, "left", "right", "up", "down"

    Returns:
            gridworld
    """
    grid = np.zeros((size, size))
    if color:
        grid[:-1, 0] = 4
        grid[1:, -1] = 2
        grid[0, 1:] = 5
        grid[-1, :-1] = 3
        grid = generate_random_barriers_(grid, N=N, max_steps=max_steps, color=1, align=align, width=width)
    else:
        grid[:, 0] = 1
        grid[:, -1] = 1
        grid[0, :] = 1
        grid[-1, :] = 1
        grid = generate_random_barriers_(grid, N=N, max_steps=max_steps, color=1, align=align, width=width)

    if reward == "rand_corner":
        i = np.random.randint(0, 4)
        reward = ["tl", "tr", "bl", "br"][i]

    if reward == "tl":
        grid[1, 1] = -1
    elif reward == "tr":
        grid[1, -2] = -1
    elif reward == "bl":
        grid[-2, 1] = -1
    elif reward == "br":
        grid[-2, -2] = -1
    return grid.astype(float)

def generate_new_reward(gridworld):
    grid = gridworld.copy()
    grid[grid == -1] = 0
    random_corner = ["tl", "tr", "bl", "br"][np.random.randint(0, 4)]
    if random_corner == "tl":
        grid[1, 1] = -1
    elif random_corner == "tr":
        grid[1, -2] = -1
    elif random_corner == "bl":
        grid[-2, 1] = -1
    elif random_corner == "br":
        grid[-2, -2] = -1

    return grid

def generate_random_barriers_(grid, N=4, max_steps=1000, color=1, align=False, width=[3]):
    grid2 = grid.copy()
    i = 0
    steps = 0

    if align:
        if type(align) == list:
            fixed_direction = align[np.random.randint(0, len(align))]
        else:
            fixed_direction = np.random.randint(0, 4)
    while i < N and steps < max_steps:
        steps += 1
        x = np.random.randint(1, grid.shape[0] - np.max(width))
        y = np.random.randint(1, grid.shape[1] - np.max(width))
        if not align:
            k = np.random.randint(0, 4)
        else:
            k = fixed_direction
            
        width_idx = np.random.randint(0, len(width))
        width_ = width[width_idx]
        new_grid = create_u_barrier(grid2, x, y, width_, k, color)
        if new_grid is not None:
            grid2 = new_grid
            i += 1
        else:
            continue
        
    return grid2


grid_rep_color_4 = np.array([[ 4.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
         5.,  5.,  5.,  5.,  5.,  5.,  5.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         1.,  1.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,
         1.,  1.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
         1.,  1.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
         1.,  1.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
         1.,  1.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
         1.,  0.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,
         1.,  0.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0., -1.,  2.],
       [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
         3.,  3.,  3.,  3.,  3.,  3.,  2.]])

grid_rep_color_5 = np.array([[ 4.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
         5.,  5.,  5.,  5.,  5.,  5.,  5.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0., -1.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,
         1.,  0.,  1.,  0.,  1.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,
         1.,  0.,  1.,  0.,  1.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,
         1.,  0.,  1.,  1.,  1.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,
         1.,  1.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,
         1.,  1.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,
         1.,  0.,  1.,  1.,  1.,  0.,  2.],
       [ 4.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,
         1.,  0.,  1.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,
         1.,  0.,  1.,  1.,  1.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
         3.,  3.,  3.,  3.,  3.,  3.,  2.]])

grid_rep_color_6 = np.array([[ 4.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
         5.,  5.,  5.,  5.,  5.,  5.,  5.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,
         1.,  1.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,
         0.,  1.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,
         0.,  1.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,
         0.,  1.,  1.,  1.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,
         0.,  0.,  0.,  1.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  1.,  1.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,
         0.,  1.,  0.,  1.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,
         0.,  1.,  0.,  1.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  1.,  1.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0., -1.,  2.],
       [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
         3.,  3.,  3.,  3.,  3.,  3.,  2.]])

two_color_wall_grid = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2,],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]]
    )

def generate_random_connected_grid(size, reward="br", fill_ratio=0.5):
    """
    Generate a gridworld where all empty spaces are connected.
    Walls can be isolated islands.

    Parameters:
    - N: Number of rows
    - M: Number of columns

    Returns:
    - grid: A 2D list representing the gridworld
    """
    
    N, M = size-1, size-1
    # Initialize grid with walls (1)
    grid = [[1 for _ in range(M)] for _ in range(N)]
    total_cells = (N-1) * (M-1)
    target_empty_cells = max(1, int(total_cells * fill_ratio))
    empty_cells = 1  # Start with one empty cell
    
    if reward == "rand":

        # Random starting point
        start_x = random.randint(1, N - 1)
        start_y = random.randint(1, M - 1)
    elif reward == "br":
        start_x, start_y = N - 3, M - 3
    elif reward == "bl":
        start_x, start_y = N - 3, 1
    elif reward == "tr":
        start_x, start_y = 1, M - 3
    elif reward == "tl":
        start_x, start_y = 1, 1
    
    grid[start_x][start_y] = 0  # Mark starting cell as empty

    # List of cells to process
    cells_to_process = [(start_x, start_y)]

    while cells_to_process and empty_cells < target_empty_cells:
        x, y = cells_to_process.pop(random.randint(0, len(cells_to_process) - 1))

        # Randomly shuffle directions
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < N-1 and 1 <= ny < M-1 and grid[nx][ny] == 1:
                # Randomly decide whether to turn it into empty space
                if random.random() < fill_ratio:
                    grid[nx][ny] = 0
                    empty_cells += 1
                    cells_to_process.append((nx, ny))
                # We stop expanding if we've reached the desired number of empty cells
                if empty_cells >= target_empty_cells:
                    print(empty_cells)
                    break

    grid = np.array(grid)
    grid[start_x][start_y] = -1
    return grid


def _resolve_reward(size, reward):
    """Return the reward coordinate and the resolved corner label."""
    max_idx = size - 2
    corner_positions = {
        "tl": (1, 1),
        "tr": (1, max_idx),
        "bl": (max_idx, 1),
        "br": (max_idx, max_idx),
    }

    if reward == "rand_corner":
        reward = random.choice(list(corner_positions.keys()))

    if reward == "rand":
        return (random.randint(1, max_idx), random.randint(1, max_idx)), "rand"

    resolved_label = reward if reward in corner_positions else "br"
    return corner_positions[resolved_label], resolved_label


def _choose_start_coordinate(size, reward_label):
    """Pick a default start location that is opposite of the reward."""
    corner_positions = {
        "tl": (1, 1),
        "tr": (1, size - 2),
        "bl": (size - 2, 1),
        "br": (size - 2, size - 2),
    }
    opposite_corners = {
        "tl": "br",
        "tr": "bl",
        "bl": "tr",
        "br": "tl",
    }
    if reward_label == "rand":
        return 1, 1
    start_label = opposite_corners.get(reward_label, "tl")
    return corner_positions[start_label]


def _passable(grid):
    return {
        (r, c)
        for r in range(1, grid.shape[0] - 1)
        for c in range(1, grid.shape[1] - 1)
        if grid[r, c] != 1
    }


def _reachable_from(grid, start):
    rows, cols = grid.shape
    queue = deque([start])
    visited = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < rows
                and 0 <= ny < cols
                and grid[nx, ny] != 1
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


def _ensure_reward_cell(grid, reward):
    rx, ry = reward
    grid[rx, ry] = 0
    grid[rx, ry] = -1


def _carve_straight_path(grid, start, goal):
    """Guarantee a simple Manhattan path between two cells."""

    x, y = start
    gx, gy = goal
    step_x = 1 if gx > x else (-1 if gx < x else 0)
    step_y = 1 if gy > y else (-1 if gy < y else 0)

    while x != gx:
        x += step_x
        grid[x, y] = 0
    while y != gy:
        y += step_y
        grid[x, y] = 0


def generate_depth_first_maze(size, reward="br", loop_probability=0.05):
    """Generate a depth-first search maze with optional extra loops."""

    grid = np.ones((size, size), dtype=int)
    interior_rows, interior_cols = size - 2, size - 2

    if interior_rows % 2 == 0:
        interior_rows -= 1
    if interior_cols % 2 == 0:
        interior_cols -= 1

    start = (1, 1)
    stack = [start]
    visited = {start}
    grid[start] = 0

    def _neighbors(cell):
        x, y = cell
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= interior_rows and 1 <= ny <= interior_cols and (nx, ny) not in visited:
                yield nx, ny

    while stack:
        current = stack[-1]
        neighbors = list(_neighbors(current))
        if neighbors:
            nx, ny = random.choice(neighbors)
            visited.add((nx, ny))
            stack.append((nx, ny))
            wall_x = current[0] + (nx - current[0]) // 2
            wall_y = current[1] + (ny - current[1]) // 2
            grid[nx, ny] = 0
            grid[wall_x, wall_y] = 0
        else:
            stack.pop()

    possible_walls = [
        (r, c)
        for r in range(1, interior_rows + 1)
        for c in range(1, interior_cols + 1)
        if grid[r, c] == 1
    ]
    for wall in possible_walls:
        if random.random() < loop_probability:
            x, y = wall
            open_neighbors = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if grid[x + dx, y + dy] == 0:
                    open_neighbors += 1
            if open_neighbors >= 2:
                grid[x, y] = 0

    reward_coord, _ = _resolve_reward(size, reward)
    _ensure_reward_cell(grid, reward_coord)
    return grid.astype(float)


def generate_random_walk_caves(size, reward="br", walk_length=None):
    """Generate caverns by carving a random walk inside a solid block."""

    grid = np.ones((size, size), dtype=int)
    reward_coord, reward_label = _resolve_reward(size, reward)
    start = _choose_start_coordinate(size, reward_label)
    current = start
    grid[current] = 0
    if walk_length is None:
        walk_length = (size - 2) * (size - 2) * 4
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(walk_length):
        dx, dy = random.choice(directions)
        nx = min(max(current[0] + dx, 1), size - 2)
        ny = min(max(current[1] + dy, 1), size - 2)
        grid[nx, ny] = 0
        current = (nx, ny)

    grid[start] = 0
    reachable = _reachable_from(grid, start)
    if reward_coord not in reachable:
        _carve_straight_path(grid, start=start, goal=reward_coord)
    _ensure_reward_cell(grid, reward_coord)
    return grid.astype(float)


def generate_scattered_obstacles_grid(size, reward="br", obstacle_density=0.2, max_attempts=5000):
    """Generate a connected open room with randomly scattered obstacles."""

    grid = np.ones((size, size), dtype=int)
    interior = size - 2
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            grid[r, c] = 0

    reward_coord, reward_label = _resolve_reward(size, reward)
    start = _choose_start_coordinate(size, reward_label)
    passable_target = interior * interior
    obstacles_to_place = int(passable_target * obstacle_density)
    placed = 0
    attempts = 0

    while placed < obstacles_to_place and attempts < max_attempts:
        attempts += 1
        r = random.randint(1, size - 2)
        c = random.randint(1, size - 2)
        if (r, c) in {reward_coord, start} or grid[r, c] == 1:
            continue
        grid[r, c] = 1
        reachable = _reachable_from(grid, start)
        if reward_coord not in reachable or len(reachable) != len(_passable(grid)):
            grid[r, c] = 0
        else:
            placed += 1

    _ensure_reward_cell(grid, reward_coord)
    return grid.astype(float)
