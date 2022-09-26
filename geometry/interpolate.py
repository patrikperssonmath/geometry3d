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

        shape = img.shape

        if len(shape)==5:
            img = img.view(-1, img.shape[-3], img.shape[-2], img.shape[-1])
            x_cord = x_cord.view(-1, x_cord.shape[-3], x_cord.shape[-2], x_cord.shape[-1])
            
        x_cord = 2.0*x_cord[:, 0:2, :, :].permute(0, 2, 3, 1) - 1.0

        img_w = torch.nn.functional.grid_sample(img, x_cord,
                                               align_corners=True,
                                               padding_mode="border",
                                               mode=self.mode)

        if len(shape)==5:
            B,N,_,_,_ = shape

            img_w = img_w.view(B, N, img_w.shape[-3], img_w.shape[-2], img_w.shape[-1])

        return img_w

