""" warps image from target to source """

import torch
from torch import jit
from geometry.transform import TransformLayer
from geometry.interpolate import Interpolate

class Warp(jit.ScriptModule):
    """ warps image from target to source """

    def __init__(self, W, H, mode="bilinear"):
        super().__init__()

        self.transform = TransformLayer(W, H)

        self.interpolate = Interpolate(mode)

        self.register_buffer("zero", torch.tensor([0.0], dtype=torch.float32).squeeze(0))

    @jit.script_method
    def forward(self, img, inv_depth, transform, calib, divison_lambda, non_rigid: bool):
        """ call function """

        shape = inv_depth.shape

        if len(shape)==5:
            img = img.view(-1, img.shape[-3], img.shape[-2], img.shape[-1])
            inv_depth = inv_depth.view(-1, 1, inv_depth.shape[-2], inv_depth.shape[-1])

            if non_rigid:
                transform = transform.view(-1, transform.shape[2], transform.shape[3], 4, 4)
            else:
                transform = transform.view(-1, 4, 4)

            calib = calib.view(-1, 4)
            divison_lambda = divison_lambda.view(-1, 1)

        x_proj, mask = self.transform.forward(inv_depth, transform, calib, divison_lambda, non_rigid)

        img_w = self.interpolate.forward(img, x_proj).where(mask, self.zero)

        if len(shape)==5:
            B,N,_,_,_ = shape

            img_w = img_w.view(B, N, img_w.shape[-3], img_w.shape[-2], img_w.shape[-1])

            mask = mask.view(B, N, 1, img_w.shape[-2], img_w.shape[-1])

        return img_w, mask, x_proj
