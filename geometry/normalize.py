""" normalizes pixel coordinates and applies undistortion by the division model"""
import torch

from geometry.utility import apply_matrix, apply_inverse_calibration
from torch import jit

class Normalize(jit.ScriptModule):
    """ normalizes pixel coordinates and applies undistortion by the division model"""

    def __init__(self, distortion=True):
        super().__init__()
        self.distortion = distortion

        self.register_buffer("one", torch.tensor(
            [1.0], dtype=torch.float32).squeeze(), persistent=True)

    @jit.script_method
    def forward(self, grid, calib, divison_lambda):
        """ call function """

        if self.distortion:

            grid = apply_inverse_calibration(grid, calib)

            grid, valid = self.apply_undistortion(grid, divison_lambda)

            return grid, valid

        else:

            return apply_inverse_calibration(grid, calib), self.one > 0

    @jit.script_method
    def apply_undistortion(self, grid, lam_d):
        """ applies undistortion """

        lam_d = lam_d.view(-1, 1, 1, 1)

        r_d = grid[:, 0:2]

        factor = (self.one + lam_d * r_d.square().sum(dim=1, keepdim=True))

        grid_out = r_d / factor

        grid_out = torch.cat((grid_out, grid[:, 2:]), dim=1)

        return grid_out, factor > 0
