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


