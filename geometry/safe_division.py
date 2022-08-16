""" divides by the last coordinate and masks divison by zero """
import torch
from torch import jit


class SafeDivision(jit.ScriptModule):
    """ divides by the last coordinate and masks divison by zero """

    def __init__(self):
        super().__init__()

        self.register_buffer("one", torch.tensor(
            [1.0], dtype=torch.float32).squeeze(), persistent=True)

    @jit.script_method
    def forward(self, x, y):
        """ call function """

        mask_src = y.abs() > 0

        x_proj = x/y.where(mask_src, self.one)

        return x_proj, mask_src
