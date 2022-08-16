""" contains functions for geometric manipulation """

import torch

@torch.jit.script
def to_homogeneous(grid, d_inv):
    """ create homogeneous vector with inverse depth as the last coordiante """

    return torch.cat((grid.expand(d_inv.shape[0], -1, -1, -1), d_inv), dim=1)


@torch.jit.script
def to_intrinsic_mat(calibration: torch.Tensor):
    """ converts calibration vector to calibration matrix"""

    batch, _ = calibration.size()

    f_x, f_y, c_x, c_y = calibration.unbind(1)

    k_mat = calibration.new_zeros((batch, 4, 4))

    k_mat[:, 0, 0] = f_x
    k_mat[:, 1, 1] = f_y
    k_mat[:, 0, 2] = c_x
    k_mat[:, 1, 2] = c_y
    k_mat[:, 2, 2] = 1.0
    k_mat[:, 3, 3] = 1.0

    return k_mat


@torch.jit.script
def to_intrinsic_mat_inv(calibration: torch.Tensor):
    """ converts calibration vector to inverse calibration matrix"""

    batch, _ = calibration.size()

    f_x, f_y, c_x, c_y = calibration.unbind(1)

    k_inv_mat = calibration.new_zeros((batch, 4, 4))

    k_inv_mat[:, 0, 0] = 1.0/f_x
    k_inv_mat[:, 1, 1] = 1.0/f_y
    k_inv_mat[:, 0, 2] = -c_x/f_x
    k_inv_mat[:, 1, 2] = -c_y/f_y
    k_inv_mat[:, 2, 2] = 1.0
    k_inv_mat[:, 3, 3] = 1.0

    return k_inv_mat


def create_grid(width, height, device=None):
    """ creates a grid and normalizes coordiantes to [0, 1] """

    grid_x, grid_y = torch.meshgrid(torch.arange(0, width, dtype=torch.float32,
                                                 device=device),
                                    torch.arange(0, height, dtype=torch.float32,
                                                 device=device),
                                    indexing="xy")

    grid_x = torch.unsqueeze(grid_x, 0)/(width-1.0)

    grid_y = torch.unsqueeze(grid_y, 0)/(height-1.0)

    grid = torch.cat((grid_x, grid_y, torch.ones_like(grid_x)), dim=0)

    grid = torch.unsqueeze(grid, 0)

    return grid


@torch.jit.script
def apply_matrix(mat, vec: torch.Tensor):
    """ multiplies mat with X of the size B, C, H, W """

    B, C, H, W = vec.size()

    vec = mat @ vec.view(B, C, H * W)

    return vec.view(B, C, H, W)
