# -*- coding: utf-8 -*-

import math

import torch
from torch import nn, Tensor, Size
from torch.distributions.gamma import Gamma

from abc import abstractmethod

import logging

logger = logging.getLogger(__name__)


class BaseNoiseDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self,
               shape: Size) -> Tensor:
        raise NotImplementedError


class SumOfGammaNoiseDistribution(BaseNoiseDistribution):
    def __init__(self,
                 k: float,
                 nb_iterations: int = 10):
        super().__init__()
        self.k = k
        self.nb_iterations = nb_iterations

    def sample(self,
               shape: Size) -> Tensor:
        samples = torch.zeros(size=shape)
        for i in range(1, self.nb_iterations + 1):
            gamma = Gamma(1. / self.k, i / self.k)
            samples = samples + gamma.sample(sample_shape=shape)
        samples = (samples - math.log(self.nb_iterations)) / self.k
        return samples
