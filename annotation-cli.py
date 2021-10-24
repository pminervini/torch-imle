#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np

import torch
from torch import nn, Tensor

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

from solvers.dijkstra import get_solver


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


def main(argv):
    neighbourhood_fn = "8-grid"
    solver = get_solver(neighbourhood_fn)

    def torch_solver(weights_batch: Tensor) -> Tensor:
        print('YYY', weights_batch.shape)
        weights_batch = weights_batch.detach().cpu().numpy()
        y_batch = np.asarray([solver(w) for w in list(weights_batch)])
        return torch.tensor(y_batch, requires_grad=False)

    with torch.inference_mode():
        weights_1 = np.ones(shape=[8, 8], dtype=float)
        weights_2 = np.ones(shape=[8, 8], dtype=float)
        weights_2[1:4, 0] = 0
        weights_2[3, 0:3] = 0

        weights_1_batch = torch.tensor(weights_1).unsqueeze(0)
        weights_2_batch = torch.tensor(weights_2).unsqueeze(0)

        y_1_batch = torch_solver(weights_1_batch)
        y_2_batch = torch_solver(weights_2_batch)

    loss_fn = HammingLoss()

    weights_1 = np.ones(shape=[1, 8, 8], dtype=float)
    weights_1_tensor = torch.tensor(weights_1)
    weights_1_params = nn.Parameter(weights_1_tensor, requires_grad=True)

    y_2_tensor = torch.tensor(y_2_batch.detach().cpu().numpy())

    target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
    noise_distribution = SumOfGammaNoiseDistribution(k=8.0 * 1.3, nb_iterations=100)

    imle_solver = imle(torch_solver,
                       target_distribution=target_distribution,
                       noise_distribution=noise_distribution,
                       nb_samples=1)

    imle_y_tensor = imle_solver(weights_1_params)

    print(imle_y_tensor.shape, y_2_batch.shape)

    loss = loss_fn(imle_y_tensor, y_2_tensor)

    print(loss)

    loss.backward()


if __name__ == '__main__':
    main(sys.argv[1:])
