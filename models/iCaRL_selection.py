import torch

from config import Config
from models import iCaRL
from models.utils.selectors import SelectorBuilder


class iCaRLSelection(iCaRL):

    def __init__(self, loss, resnet_type="32", num_classes=10, k=2000, selector='l2', **selector_params):
        super(iCaRLSelection, self).__init__(loss, resnet_type, num_classes, k)

        self.nearest_selector = self.get_selector_config(selector, selector_params)

    @staticmethod
    def get_selector_config(selector, selector_params):
        return SelectorBuilder.build(selector, **selector_params)

    def _nearest_prototype(self, centers, features):
        return self.nearest_selector.get_nearest_centers(centers, features)

    def _get_closest_feature(self, center, features):
        return self.nearest_selector.get_nearest_feature(center, features)

    def _extract_features_and_mean(self, dataloader):
        features = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(Config.DEVICE)
                #features.append(self.nearest_selector.metric.normalize(self.features_extractor(images), dim=1))
                features.append(self.features_extractor(images))

            features = torch.cat(features)
            mean = features.mean(dim=0)

        #return features, self.nearest_selector.metric.normalize(mean)
        return features, mean
