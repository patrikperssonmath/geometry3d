""" projects 3d points to image plane"""

import torch
from torch import jit


class ViewMasker(jit.ScriptModule):
    """ projects 3d points to image plane"""

    def __init__(self):
        super().__init__()

        self.register_buffer("dim_upper", tensor=torch.tensor(
            [1.0, 1.0], dtype=torch.float32).view(1, 2, 1, 1), persistent=True)

        self.register_buffer("dim_lower", tensor=torch.tensor(
            [0.0, 0.0], dtype=torch.float32).view(1, 2, 1, 1), persistent=True)

    @jit.script_method
    def forward(self, x_projection):
        """ createse mask of visible projections """

        x_projection = x_projection[:, 0:2]

        return torch.logical_and(x_projection <= self.dim_upper,
                                 x_projection >= self.dim_lower).all(1, keepdim=True)
