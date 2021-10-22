# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor, Size

from imle.distributions import BaseNoiseDistribution, SumOfGammaNoiseDistribution

from typing import Callable

import logging

logger = logging.getLogger(__name__)


def wrapped(function: Callable[[Tensor], Tensor] = None,
            noise_distribution: BaseNoiseDistribution = None,
            nb_samples: int = 1,
            lambda_: float = 1.0):

    if noise_distribution is None:
        noise_distribution = SumOfGammaNoiseDistribution(k=20.0, nb_iterations=100)

    if function is None:
        return functools.partial(wrapped, nb_samples=nb_samples, noise_distribution=noise_distribution)

    @functools.wraps(function)
    def wrapper(input: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input: Tensor, *args):
                input_shape = input.shape
                perturbed_input_shape = Size([nb_samples] + list(input_shape))

                noise = noise_distribution.sample(shape=perturbed_input_shape)
                perturbed_input = input.unsqueeze(0) + noise

                perturbed_output = function(perturbed_input)

                ctx.save_for_backward(input, noise, perturbed_output)

                res = perturbed_output.mean(0)
                return res

            @staticmethod
            def backward(ctx, dy):
                input, noise, perturbed_output = ctx.saved_variables

                target_input = input + lambda_ * dy
                target_perturbed_input = target_input.unsqueeze(0) + noise

                target_output = function(target_perturbed_input)

                res = perturbed_output.mean(0) - target_output.mean(0)
                return res
