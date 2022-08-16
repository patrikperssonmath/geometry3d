""" divides by the depth and masks divison by zero or negative number 
        returns projection and inverse depth
    """

import torch
from torch import jit


class Project(jit.ScriptModule):
    """ divides by the depth and masks divison by zero or negative number
    """

    def __init__(self):
        super().__init__()

        self.register_buffer("one", torch.tensor(
            [1.0], dtype=torch.float32).squeeze(), persistent=True)

    @jit.script_method
    def forward(self, X):
        """ call function """

        depth = X[:, 2:3]

        mask_src = depth > 0

        x_proj = X/depth.where(mask_src, self.one)

        return x_proj, mask_src
