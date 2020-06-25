import torch
from models.utils import l2_normalize
from models.utils import metrics


class Selector:

    def __init__(self, metric):
        self.metric = metric

    def calculate_distances(self, centers, features):
        return self.metric.calculate_distance(centers, features)

    def get_nearest_centers(self, centers, features):
        """
        :param centers: A tensor embedding all the centers.
        :param features: A tensor embedding a batch of features.
        :return: the index of the nearest center for each tensor.
        """
        centers, norm_features = Selector.prepare(centers, features)
        _, preds = self.calculate_distances(centers, norm_features).min(1)
        return preds

    def get_nearest_feature(self, center, features):
        """
        :param center: A tensor that represents a center.
        :param features:  tensor embedding a batch of features.
        :return: the index of the nearest feature for a center.
        """
        center = center.unsqueeze(0)
        center, norm_features = Selector.prepare(center, features)
        closest_idx = self.calculate_distances(center, norm_features).argmin().item()
        return closest_idx

    @staticmethod
    def prepare(centers, features):
        # It is important to keep track of the dimensions of the tensors to calculate properly the nearest one.
        centers = torch.stack([centers] * features.shape[0])            # (batch_size, n_classes, feature_size)
        centers = centers.transpose(1, 2)                               # (batch_size, feature_size, n_classes)
        norm_features = l2_normalize(features, dim=1).unsqueeze(2)      # (batch_size, feature_size, 1)
        return centers, norm_features


class SelectorBuilder:

    @staticmethod
    def build(metric, **kwargs):
        metric = SelectorBuilder.get_metric(metric, **kwargs)
        return Selector(metric)

    @staticmethod
    def get_metric(metric, **params):
        metric = metric.lower()
        """
        Insert more metrics here. If some params are required it is possible to send them as following:
            if metric == 'Example':
                return Example(params)

        and params is the dict that is sent to the builder class.
        """
        if metric == 'l1':
            return metrics.L1Metric()
        elif metric == 'l2':
            return metrics.L2Metric()
        elif metric == 'canberra':
            return metrics.CanberraMetric()
        elif metric == 'minkowski':
            return metrics.MinkowskiMetric(**params)
        elif metric == 'cos_sim':
            return metrics.CosineMetric()
        elif metric == 'cheb':
            return metrics.ChebyshevMetric()
        else:
            raise NotImplementedError('Metric not implemented.')


