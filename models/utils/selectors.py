from abc import ABC, abstractmethod

import torch

from models.utils import l2_normalize


class Selector(ABC):

    @abstractmethod
    def get_nearest_centers(self, centers, features):
        """
        :param centers: A tensor embedding all the centers.
        :param features: A tensor embedding a batch of features.
        :return: the index of the nearest center for each tensor.
        """
        pass

    @abstractmethod
    def get_nearest_feature(self, center, features):
        """
        :param center: A tensor that represents a center.
        :param features:  tensor embedding a batch of features.
        :return: the index of the nearest feature for a center.
        """
        pass


class L2Selector(Selector):
    """
    The selection methods have to extend the class Selector and implement
    both abstract methods: get_nearest_centers and get_nearest_feature.
    """

    @staticmethod
    def l2_distance(centers, features):
        # It is important to keep track of the dimensions of the tensors to calculate properly the nearest one.
        centers = torch.stack([centers] * features.shape[0])            # (batch_size, n_classes, feature_size)
        centers = centers.transpose(1, 2)                               # (batch_size, feature_size, n_classes)
        norm_features = l2_normalize(features, dim=1).unsqueeze(2)      # (batch_size, feature_size, 1)

        l2_distance = torch.pow(centers - norm_features, 2).sum(1)      # in the difference of tensors if one of them
                                                                        # have a less dimension it is promoted.
        return l2_distance

    def get_nearest_centers(self, centers, features):
        _, preds = self.l2_distance(centers, features).min(1)
        return preds

    def get_nearest_feature(self, center, features):
        center = center.unsqueeze(0)
        closest_idx = self.l2_distance(center, features).argmin().item()
        return closest_idx

    """
    Other forms to implement it:
    def get_nearest_centers(self, centers, features):
        normalized_features = l2_normalize(features, dim=1)
        distances = torch.pow(center - normalized_features, 2).sum(-1)
        return distances.argmin().item()

    Or also iterating the features
    def get_nearest_feature(self, center, features):
        pred_labels = []

        for feature in features:
            norm_feature = l2_normalize(feature)
            distances = torch.pow(centers - norm_feature, 2).sum(-1)
            pred_labels.append(distances.argmin().item())

        return torch.from_numpy(np.array(pred_labels))

    """

