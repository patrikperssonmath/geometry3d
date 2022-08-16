""" base interpolation class """
import torch
from torch import jit


class Interpolate(jit.ScriptModule):
    """ base interpolation class """

    def __init__(self, mode):
        super().__init__()

        self.mode = mode

    @jit.script_method
    def forward(self, img, x_cord):
        """ interpolates coordinates x_coord: [B,2,H,W] \in [0, 1] """

        x_cord = 2.0*x_cord[:, 0:2, :, :].permute(0, 2, 3, 1) - 1.0

        return torch.nn.functional.grid_sample(img, x_cord,
                                               align_corners=True,
                                               padding_mode="border",
                                               mode=self.mode)
