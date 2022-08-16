""" maps the pixel locations in in source to target """
from torch import jit

from .normalize import Normalize
from .project import Project
from .unnormalize import Unnormalize
from .utility import apply_matrix, create_grid, to_homogeneous, to_intrinsic_mat, to_intrinsic_mat_inv
from .view_masker import ViewMasker


class TransformLayer(jit.ScriptModule):
    """ maps the pixel locations in in source to target """

    def __init__(self, W, H):
        super().__init__()

        self.unnormalzie = Unnormalize()
        self.normalize = Normalize()
        self.project = Project()
        self.view_masker = ViewMasker()

        self.register_buffer("grid", create_grid(W, H))

    @jit.script_method
    def forward(self, inv_depth, transform, calib, divison_lambda, non_rigid: bool):
        """ call function """

        K = to_intrinsic_mat(calib)

        Kinv = to_intrinsic_mat_inv(calib)

        grid, valid_norm = self.normalize(self.grid, Kinv, divison_lambda)

        X = to_homogeneous(grid, inv_depth)

        if non_rigid:

            X = transform@X.permute(0, 2, 3, 1).contiguous().unsqueeze(-1)

            X = X.squeeze(-1).permute(0, 3, 1, 2).contiguous()

        else:

            X = apply_matrix(transform, X)

        x_proj, mask_src = self.project(X)

        x_proj, valid_un_norm = self.unnormalzie(x_proj, K, divison_lambda)

        mask_src = mask_src.logical_and(
            self.view_masker.forward(x_proj[:, 0:2]))

        mask_src = mask_src.logical_and(valid_norm).logical_and(valid_un_norm)

        return x_proj, mask_src
