import torch
import context
import argparse
from geometry.transform import TransformLayer
from torch.profiler import (ProfilerActivity, profile, record_function,
                            tensorboard_trace_handler)
import os
import shutil


def profile_transform():

    path = "./profile/profiler_out"

    if os.path.exists(path):
        shutil.rmtree(path)

    device = "cuda:0"

    H = 224

    W = 224

    B = 32

    depth_inv = torch.rand(
        (B, 1, H, W), device=device, dtype=torch.float32)

    T = torch.randn((B, 4, 4), device=device, dtype=torch.float32)

    calib = torch.randn((B, 4), device=device, dtype=torch.float32)

    division_lambda = -0.01 * \
        torch.rand((B, 1), device=device, dtype=torch.float32)

    transform = TransformLayer(W,H).to("cuda:0")

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=3,
            warmup=5,
            active=10,
            repeat=1),
        on_trace_ready=tensorboard_trace_handler(path),
        with_stack=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as profiler:

        for step in range(40):
            print("step:{}".format(step))

            test = transform.forward( depth_inv, T, calib, division_lambda)

            profiler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    try:
        profile_transform()
    except KeyboardInterrupt:
        print("\nInterrupt!")
