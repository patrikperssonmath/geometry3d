""" warps image from target to source """

import torch
from torch import jit
from geometry.transform import TransformLayer
from geometry.interpolate import Interpolate

class Warp(jit.ScriptModule):
    """ warps image from target to source """

    def __init__(self, mode="bilinear"):
        super().__init__()

        self.transform = TransformLayer()

        self.interpolate = Interpolate(mode)

        self.register_buffer("zero", torch.tensor([0.0], dtype=torch.float32).squeeze(0))

    @jit.script_method
    def forward(self, img, inv_depth, transform, calib, divison_lambda, non_rigid: bool):
        """ call function """

        x_proj, mask = self.transform.forward(inv_depth, transform, calib, divison_lambda, non_rigid)

        return self.interpolate.forward(img, x_proj).where(mask, self.zero)
