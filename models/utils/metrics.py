from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class Metric(ABC):

    @abstractmethod
    def normalize(self, tensor, dim=0):
        pass

    @abstractmethod
    def norm(self, tensor, dim=0):
        pass

    @abstractmethod
    def calculate_distance(self, tensor1, tensor2):
        pass


class L1Metric(Metric):

    def normalize(self, tensor, dim=0):
        return F.normalize(tensor, p=1, dim=dim)

    def norm(self, tensor, dim=0):
        return torch.norm(tensor, p=1, dim=dim)

    def calculate_distance(self, tensor1, tensor2):
        l1_distance = self.norm(tensor1 - tensor2, dim=1)
        return l1_distance


class L2Metric(Metric):

    def normalize(self, tensor, dim=0):
        return F.normalize(tensor, p=2, dim=dim)

    def norm(self, tensor, dim=0):
        return torch.norm(tensor, p=2, dim=dim)

    def calculate_distance(self, tensor1, tensor2):
        l2_distance = self.norm(tensor1 - tensor2, dim=1)
        return l2_distance


class ChebyshevMetric(Metric):

    def normalize(self, tensor, dim=0):
        return F.normalize(tensor, p=0, dim=dim)

    def norm(self, tensor, dim=0):
        return torch.norm(tensor, p=0, dim=dim)

    def calculate_distance(self, tensor1, tensor2):
        chebyshev_distance = self.norm(tensor1 - tensor2, dim=1)
        return chebyshev_distance


class MinkowskiMetric(Metric):

    def __init__(self, p=2):
        self.p = p

    def normalize(self, tensor, dim=0):
        return F.normalize(tensor, p=self.p, dim=dim)

    def norm(self, tensor, dim=0):
        return torch.norm(tensor, p=self.p, dim=dim)

    def calculate_distance(self, tensor1, tensor2):
        minkowski_distance = self.norm(tensor1 - tensor2, dim=1)
        return minkowski_distance


class CosineMetric(Metric):

    def normalize(self, tensor, dim=0):
        return L2Metric().normalize(tensor, dim=dim)

    def __init__(self, dim=1):
        self.cos_sim = nn.CosineSimilarity(dim=dim, eps=1e-6)

    def norm(self, tensor, dim=0):
        # Cosine distance is invariant respect to the l2 normalization
        return L2Metric().norm(tensor, dim=dim)

    def calculate_distance(self, tensor1, tensor2):
        # cosine_distance = 1 - cosine_similarity
        cosine_sim = self.cos_sim(tensor1, tensor2)
        ones = torch.ones_like(cosine_sim)
        cosine_distance = ones - cosine_sim
        return cosine_distance


"""
# TODO: not working this way
class CanberraMetric(Metric):

    def norm(self, tensor, dim=0):
        return (torch.abs(tensor) / (torch.abs(tensor) + torch.abs(tensor))).sum(1)

    def calculate_distance(self, tensor1, tensor2):
        canberra_distance = self.norm(tensor1 - tensor2, dim=1)
        return canberra_distance
"""

