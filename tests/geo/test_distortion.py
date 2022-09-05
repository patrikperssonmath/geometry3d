import numpy as np
import random

import context

import torch

from geometry.normalize import Normalize

from geometry.unnormalize import Unnormalize


def test_distortion_normalize():

    x = np.linspace(-1, 1, 1000)

    y = np.linspace(-1, 1, 1000)

    xv, yv = np.meshgrid(x, y, indexing='xy')

    xv = np.expand_dims(xv, axis=0)
    yv = np.expand_dims(yv, axis=0)

    grid = np.concatenate((xv, yv))

    radial_grid = np.linalg.norm(grid, axis=0, keepdims=True)

    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
    calib = torch.tensor([1,1,0,0], dtype=torch.float32).unsqueeze(0)

    grid_tensor = torch.cat(
        (grid_tensor, torch.ones_like(grid_tensor[:, 0:1]), torch.ones_like(grid_tensor[:, 0:1])), dim=1)

    normalize = Normalize()

    for i in range(100):

        lambda_dist = - 0.2*random.random()

        p = 1/(1 + lambda_dist*np.square(radial_grid))*grid

        lambda_dist_tensor = torch.tensor(
            lambda_dist, dtype=torch.float32).unsqueeze(0)

        p_tensor, valid = normalize.forward(
            grid_tensor, calib, lambda_dist_tensor)

        p_tensor = p_tensor.squeeze().cpu()
        p_gt = torch.tensor(p, dtype=torch.float32)

        diff = torch.max((p_tensor[0:2]-p_gt).abs().masked_select(valid))

        assert diff < 5e-7

    print(grid)


def test_distortion_unnormalize():

    x = np.linspace(-1, 1, 1000)

    y = np.linspace(-1, 1, 1000)

    xv, yv = np.meshgrid(x, y, indexing='xy')

    xv = np.expand_dims(xv, axis=0)
    yv = np.expand_dims(yv, axis=0)

    grid = np.concatenate((xv, yv))

    radial_grid = np.linalg.norm(grid, axis=0, keepdims=True)

    calib = torch.tensor([1,1,0,0], dtype=torch.float32).unsqueeze(0)

    unnormalize = Unnormalize()

    for i in range(100):

        lambda_dist = - 0.4*random.random()

        p = 1/(1 + lambda_dist*np.square(radial_grid))*grid

        lambda_dist_tensor = torch.tensor(
            lambda_dist, dtype=torch.float32).unsqueeze(0)

        p = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

        p = torch.cat((p, torch.ones_like(p[:, 0:2])), dim=1)

        p_gt = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        p_gt = torch.cat((p_gt, torch.ones_like(p[:, 0:2])), dim=1)

        p_tensor, valid = unnormalize.forward(p, calib, lambda_dist_tensor)

        diff = torch.max((p_tensor-p_gt).abs().masked_select(valid))

        assert diff < 5e-7

    print(grid)


def test_distortion_unnormalize_zero():

    x = np.linspace(-1, 1, 1000)

    y = np.linspace(-1, 1, 1000)

    xv, yv = np.meshgrid(x, y, indexing='xy')

    xv = np.expand_dims(xv, axis=0)
    yv = np.expand_dims(yv, axis=0)

    grid = np.concatenate((xv, yv))

    radial_grid = np.linalg.norm(grid, axis=0, keepdims=True)

    calib = torch.tensor([1,1,0,0], dtype=torch.float32).unsqueeze(0)

    unnormalize = Unnormalize()

    for i in range(100):

        lambda_dist = 0

        p = 1/(1 + lambda_dist*np.square(radial_grid))*grid

        lambda_dist_tensor = torch.tensor(
            lambda_dist, dtype=torch.float32).unsqueeze(0)

        p = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

        p = torch.cat((p, torch.ones_like(p[:, 0:2])), dim=1)

        p_gt = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        p_gt = torch.cat((p_gt, torch.ones_like(p[:, 0:2])), dim=1)

        p_tensor, valid = unnormalize.forward(p, calib, lambda_dist_tensor)

        diff = torch.max((p_tensor-p_gt).abs().masked_select(valid))

        assert diff < 5e-7
