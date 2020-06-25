from models import iCaRL
from models.utils.selectors import L2Selector


class iCaRLSelection(iCaRL):

    def __init__(self, loss, resnet_type="32", num_classes=10, k=2000, selector='L2', **selector_params):
        super(iCaRLSelection, self).__init__(loss, resnet_type, num_classes, k)

        self.nearest_selector = self.get_selector_config(selector, selector_params)

    @staticmethod
    def get_selector_config(selector, selector_params):
        # Returns the selector
        """
        Insert more selectors here. If some params are required it is possible to send them as following:
            if selector == 'Example':
                return Example(selector_params)

        and selector_params is the dict that is sent to iCaRLSelection class.
        """
        if selector == 'L2':
            return L2Selector()
        else:
            raise NotImplementedError('Selector not implemented.')

    def _nearest_prototype(self, centers, features):
        return self.nearest_selector.get_nearest_centers(centers, features)

    def _get_closest_feature(self, center, features):
        return self.nearest_selector.get_nearest_feature(center, features)


