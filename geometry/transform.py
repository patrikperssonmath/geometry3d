""" maps the pixel locations in in source to target """
from torch import nn

from geometry.normalize import Normalize
from geometry.project import Project
from geometry.unnormalize import Unnormalize
from geometry.utility import apply_matrix, create_grid, to_homogeneous
from geometry.view_masker import ViewMasker


class TransformLayer(nn.Module):
    """ maps the pixel locations in in source to target """

    def __init__(self, W, H, distortion=True):
        super().__init__()

        self.unnormalzie = Unnormalize(distortion)
        self.normalize = Normalize(distortion)
        self.project = Project()
        self.view_masker = ViewMasker()

        self.register_buffer("grid", create_grid(W, H), persistent=False)

    def forward(self, inv_depth, transform, calib_i, divison_lambda_i, calib_j, divison_lambda_j, non_rigid: bool):
        """ call function """
       
        X = to_homogeneous(self.grid, inv_depth)

        X, valid_norm = self.normalize(X, calib_i, divison_lambda_i)

        if non_rigid:

            X = transform@X.permute(0, 2, 3, 1).contiguous().unsqueeze(-1)

            X = X.squeeze(-1).permute(0, 3, 1, 2).contiguous()

        else:

            X = apply_matrix(transform, X)

        x_proj, mask_src = self.project(X)

        x_proj, valid_un_norm = self.unnormalzie(x_proj, calib_j, divison_lambda_j)

        mask_src = mask_src.logical_and(self.view_masker.forward(x_proj[:, 0:2]))

        if valid_norm is not None:

            mask_src = mask_src.logical_and(valid_norm).logical_and(valid_un_norm)

        return x_proj, mask_src
