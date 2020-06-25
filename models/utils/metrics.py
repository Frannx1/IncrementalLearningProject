from abc import ABC, abstractmethod

import torch
from torch import nn


class Metric(ABC):

    @abstractmethod
    def norm(self, tensor):
        pass

    @abstractmethod
    def calculate_distance(self, tensor1, tensor2):
        pass


class L1Metric(Metric):

    def norm(self, tensor):
        return torch.abs(tensor).sum(1)

    def calculate_distance(self, tensor1, tensor2):
        l1_distance = self.norm(tensor1 - tensor2)
        return l1_distance


class L2Metric(Metric):

    def norm(self, tensor):
        return torch.pow(tensor, 2).sum(1)

    def calculate_distance(self, tensor1, tensor2):
        l2_distance = self.norm(tensor1 - tensor2)
        return l2_distance


class CanberraMetric(Metric):
    # TODO: not working this way

    def norm(self, tensor):
        return (torch.abs(tensor) / (torch.abs(tensor) + torch.abs(tensor))).sum(1)

    def calculate_distance(self, tensor1, tensor2):
        canberra_distance = self.norm(tensor1 - tensor2)
        return canberra_distance


class ChebyshevMetric(Metric):

    def norm(self, tensor):
        return torch.max(torch.abs(tensor), dim=1)[0]

    def calculate_distance(self, tensor1, tensor2):
        chebyshev_distance = self.norm(tensor1 - tensor2)
        return chebyshev_distance


class MinkowskiMetric(Metric):

    def __init__(self, p=2):
        self.p = p

    def norm(self, tensor):
        return torch.abs(tensor).pow(self.p).pow(1.0 / self.p).sum(1)

    def calculate_distance(self, tensor1, tensor2):
        minkowski_distance = self.norm(tensor1 - tensor2)
        return minkowski_distance


class CosineMetric(Metric):

    def __init__(self, dim=1):
        self.cos_sim = nn.CosineSimilarity(dim=dim, eps=1e-6)

    def norm(self, tensor):
        # Cosine distance is invariant respect to the l2 normalization
        return L2Metric().norm(tensor)

    def calculate_distance(self, tensor1, tensor2):
        # cosine_distance = 1 - cosine_similarity
        cosine_sim = self.cos_sim(tensor1, tensor2)
        ones = torch.ones_like(cosine_sim)
        cosine_distance = ones - cosine_sim
        return cosine_distance
