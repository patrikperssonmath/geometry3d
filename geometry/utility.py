""" contains functions for geometric manipulation """

import typing
import torch

@torch.jit.script
def se3_exp(eta:torch.Tensor):

    tx,ty,tz,wx,wy,wz = eta.unbind(-1)

    o = torch.zeros_like(wx)

    A = torch.stack([o, -wz, wy, tx, 
                    wz, o, -wx, ty, 
                    -wy, wx, o, tz,
                    o, o, o, o], dim=-1)

    if len(A.shape)==5:

        A = A.view(A.shape[0], A.shape[1], A.shape[2], A.shape[3], 4, 4)

    elif len(A.shape)==4:

        A = A.view(A.shape[0], A.shape[1], A.shape[2], 4, 4)

    elif len(A.shape)==3:

        A = A.view(A.shape[0], A.shape[1], 4, 4)

    elif len(A.shape)==2:

        A = A.view(A.shape[0], 4, 4)

    else:

        raise Exception("invalid dimensions!")

    return torch.linalg.matrix_exp(A)

@torch.jit.script
def to_homogeneous(grid, d_inv):
    """ create homogeneous vector with inverse depth as the last coordiante """

    return torch.cat((grid.expand(d_inv.shape[0], -1, -1, -1), d_inv), dim=1)

@torch.jit.script
def apply_calibration(X, calib):
    f_x, f_y, c_x, c_y = calib.view(-1,4,1,1).unbind(1)

    x, y, z, d = X.unbind(1)

    return torch.stack([f_x*x+c_x, f_y*y+c_y, z, d], dim=1)

@torch.jit.script
def apply_inverse_calibration(X, calib):
    f_x, f_y, c_x, c_y = calib.view(-1,4,1,1).unbind(1)

    x, y, z, d = X.unbind(1)

    return torch.stack([(x-c_x)/f_x, (y-c_y)/f_y, z, d], dim=1)

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

@torch.jit.script
def create_grid(width:int, height:int, dtype: torch.dtype=torch.float32, device:typing.Optional[torch.device]=None):
    """ creates a grid and normalizes coordiantes to [0, 1] """

    grid_x, grid_y = torch.meshgrid(torch.arange(0, width, dtype=dtype,
                                                 device=device),
                                    torch.arange(0, height, dtype=dtype,
                                                 device=device),
                                    indexing="xy")

    return torch.stack((grid_x/(width-1.0),
                        grid_y/(height-1.0), 
                        torch.ones_like(grid_x)), dim=0).unsqueeze(0)

@torch.jit.script
def apply_matrix(mat, vec: torch.Tensor):
    """ multiplies mat with X of the size B, C, H, W """

    B, C, H, W = vec.size()

    vec = mat @ vec.view(B, C, H * W)

    return vec.view(B, C, H, W)
