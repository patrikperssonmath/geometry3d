""" applies distortion by the division model and intrisic parameters"""
import torch
from torch import jit

from geometry.safe_division import SafeDivision
from geometry.utility import apply_matrix


class Unnormalize(jit.ScriptModule):
    """ applies distortion by the division model and intrisic parameters"""

    def __init__(self):
        super().__init__()

        self.register_buffer("one", torch.tensor(
            [1.0], dtype=torch.float32).squeeze(), persistent=True)

        self.register_buffer("zero", torch.tensor(
            [0.0], dtype=torch.float32).squeeze(), persistent=True)

        self.safe_division = SafeDivision()

    @jit.script_method
    def forward(self, x_proj_in, K, divison_lambda):
        """ call function """

        x_proj, mask, valid1 = self.apply_distortion_new(
            x_proj_in, divison_lambda)

        x_proj = torch.where(mask.logical_and(valid1), x_proj, x_proj_in)

        x_proj, valid2 = self.apply_distortion_itr(
            x_proj, x_proj_in, divison_lambda)

        return apply_matrix(K, x_proj), valid1.logical_and(valid2)

    @jit.script_method
    def apply_distortion_itr(self, x_d, x_u, lam_d):
        """
        when the second order polynomial is degenerate (always works given sufficient iterations)
        """

        lam_d = lam_d.view(-1, 1, 1, 1)

        hmg = x_u[:, 2:]

        x_d = x_d[:, 0:2]
        x_u = x_u[:, 0:2]

        b = x_u.square().sum(dim=1, keepdim=True)
        a = x_d.square().sum(dim=1, keepdim=True)

        for _ in range(2):

            a = a + (a-(1.0+lam_d*a).pow(2.0)*b)/(2*lam_d*(1.0+lam_d*a)*b-1.0)

        a = torch.where(a > self.zero, a, self.zero)

        factor = (1.0+lam_d*a)

        x_d = factor*x_u

        return torch.cat((x_d, hmg), dim=1), factor > 0

    @jit.script_method
    def apply_distortion_new(self, x_u, lam_d):
        """
        when the second order polynomial is not degenerate
        """

        lam_d = lam_d.view(-1, 1, 1, 1)

        hmg = x_u[:, 2:]

        x_u = x_u[:, 0:2]

        r_u_square = x_u.square().sum(dim=1, keepdim=True)

        tmp = lam_d.abs()*r_u_square

        mask = tmp > 1e-1

        tmp = tmp.where(mask, self.one)

        r_ur_d = (1.0-(4.0*tmp + 1.0).sqrt())/2.0

        factor = (1.0 - r_ur_d.square()/tmp)

        x_d = factor*x_u

        return torch.cat((x_d, hmg), dim=1), mask, factor > 0
