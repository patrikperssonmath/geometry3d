""" maps image from target to source """

import torch
from torch import jit
from geometry.transform import TransformLayer
from geometry.interpolate import Interpolate

class Map(jit.ScriptModule):
    """ maps image from target to source """

    def __init__(self, W, H, mode="bilinear"):
        super().__init__()

        self.transform = TransformLayer(W, H)

    @jit.script_method
    def forward(self, inv_depth, transform, calib, divison_lambda, non_rigid: bool):
        """ call function """

        shape = inv_depth.shape

        if len(shape)==5:
            inv_depth = inv_depth.view(-1, 1, inv_depth.shape[-2], inv_depth.shape[-1])

            if non_rigid:
                transform = transform.view(-1, transform.shape[2], transform.shape[3], 4, 4)
            else:
                transform = transform.view(-1, 4, 4)

            calib = calib.view(-1, 4)
            divison_lambda = divison_lambda.view(-1, 1)

        x_proj, mask = self.transform.forward(inv_depth, transform, calib, divison_lambda, non_rigid)

        if len(shape)==5:
            B,N,_,_,_ = shape

            mask = mask.view(B, N, 1, inv_depth.shape[-2], inv_depth.shape[-1])

            x_proj = x_proj.view(B, N, -1, inv_depth.shape[-2], inv_depth.shape[-1])

        return mask, x_proj
