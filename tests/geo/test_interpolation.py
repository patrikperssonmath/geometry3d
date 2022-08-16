
import context
from geometry.interpolate import Interpolate
from geometry.utility import create_grid
import torch
import numpy as np
import kornia as K
torch.backends.cuda.matmul.allow_tf32 = False

torch.backends.cudnn.allow_tf32 = False


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def test_interpolation():

    dim = 100

    image = np.random.rand(dim, dim).astype(np.float32)

    points = (dim-1)*np.random.rand(100, 2).astype(np.float32)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    image_tensor = K.filters.gaussian_blur2d(
        image_tensor, (11, 11), (1, 1))

    image = image_tensor.squeeze().numpy()

    interpolator = Interpolate("bilinear")

    point = torch.tensor(points, dtype=torch.float32).view(
        1, 1, -1, 2).permute(0, 3, 1, 2)/(dim-1)

    x, y = np.split(points, 2, axis=1)

    val_ref = bilinear_interpolate(image, x, y)

    val_test = interpolator(image_tensor, point).view(-1, 1).numpy()

    diff = np.mean(np.abs(val_ref-val_test))

    assert diff < 1e-7

    print(val_ref)
    print(val_test)


def test_identity_sampling():

    image = np.random.rand(100, 100).astype(np.float32)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    image_tensor = K.filters.gaussian_blur2d(
        image_tensor, (11, 11), (1, 1))

    grid = create_grid(100, 100)

    interpolator = Interpolate("bilinear")

    image_new = interpolator(image_tensor, grid)

    diff = (image_tensor-image_new).abs().mean()

    assert diff.item() < 1e-7
