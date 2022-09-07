import context

import lietorch
import numpy as np
import sophus as sp

import torch
from geometry.transform import TransformLayer
from geometry.utility import to_intrinsic_mat, to_intrinsic_mat_inv, create_grid, to_homogeneous, se3_exp
from geometry.view_masker import ViewMasker
from scipy.spatial.transform import Rotation
import lietorch
import lietorch

torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True


def test_transfom_mat():

    for i in range(100):
        theta = 1e-1*torch.randn((1,1,100,100, 6), dtype=torch.float32)

        Tli = lietorch.SE3.exp(theta).matrix()
        T = se3_exp(theta)

        diff = torch.max(np.abs(Tli-T))

        assert diff < 1e-4

        theta = 1e-1*torch.randn((1, 100,100, 6), dtype=torch.float32)

        Tli = lietorch.SE3.exp(theta).matrix()
        T = se3_exp(theta)

        diff = torch.max(np.abs(Tli-T))

        assert diff < 1e-4

        theta = 1e-1*torch.randn((5, 6), dtype=torch.float32)

        Tli = lietorch.SE3.exp(theta).matrix()
        T = se3_exp(theta)

        diff = torch.max(np.abs(Tli-T))

        assert diff < 1e-4

def test_transform():

    for i in range(100):

        W = H = 100

        d_inv = torch.randn((1, 1, H, W), dtype=torch.float32).abs() + 0.01

        log_pose = 1e-1*torch.randn((1, 6), dtype=torch.float32)

        calib = torch.cat((1e-1*torch.randn((1, 2), dtype=torch.float32)+1, 1e-1 *
                           torch.randn((1, 2), dtype=torch.float32)+0.5), dim=1)

        lambda_dist = torch.zeros(1, 1)

        T = lietorch.SE3.exp(log_pose).matrix()

        T_non_rigid = T#.view(1, 1, 1, 4, 4).expand(-1,H,W,-1,-1)

        transform = TransformLayer(W,H)

        x_proj, mask_src = transform.forward(
            d_inv, T_non_rigid, calib, lambda_dist)

        ###### calculate gt! ######

        K = to_intrinsic_mat(calib).cpu().numpy()

        Kinv = to_intrinsic_mat_inv(calib).cpu().numpy()

        T = T.cpu().squeeze(0).numpy()

        X = to_homogeneous(create_grid(W, H), d_inv).cpu().numpy()

        X = (K@T@Kinv)@np.reshape(X, (1, 4, H*W))

        X = np.reshape(X, (1, 4, H, W))

        X = X / X[:, 3:]

        depth = X[:, 2:3]

        x_coord = X / depth

        mask_gt = x_coord[:, 0:2] >= 0
        mask_gt = np.logical_and((x_coord[:, 0:2] <= 1), mask_gt).all(axis=1)
        mask_gt = np.logical_and(depth > 0, mask_gt)

        ##### compare #####

        assert (mask_src == torch.tensor(mask_gt, dtype=torch.float32)).all()

        diff = torch.max(
            (x_proj-torch.tensor(x_coord, dtype=torch.float32)).abs().masked_select(mask_src))

        assert diff < 1e-5


def test_view_masker():

    view_masker = ViewMasker()

    x_coord = np.random.randn(1, 2, 100, 100) + 0.5

    mask_gt = x_coord >= 0
    mask_gt = np.logical_and((x_coord <= 1), mask_gt).all(axis=1)

    x_coord = torch.tensor(x_coord, dtype=torch.float32)

    mask_pred = view_masker.forward(x_coord).cpu().numpy()

    assert (mask_pred == mask_gt).all()


def test_to_K():

    calib = np.random.randn(1, 4)

    K = to_intrinsic_mat(torch.tensor(calib)).squeeze().numpy()

    calib_new = np.array((K[0, 0], K[1, 1], K[0, 2], K[1, 2]))

    diff = np.max(np.abs(calib-calib_new))

    assert diff < 1e-7


def test_to_K_inv():

    calib = np.random.randn(1, 4)

    K = np.eye(4, 4)
    K[0, 0] = calib[0, 0]
    K[1, 1] = calib[0, 1]
    K[0, 2] = calib[0, 2]
    K[1, 2] = calib[0, 3]

    K_inv_np = np.linalg.inv(K)

    Kinv = to_intrinsic_mat_inv(torch.tensor(calib)).squeeze().numpy()

    diff = np.max(np.abs(Kinv-K_inv_np))

    assert diff < 1e-7
