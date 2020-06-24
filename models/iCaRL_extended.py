import numpy as np
import torch
from torch import nn

from config import Config
from models import iCaRL
from models.utils import l2_normalize

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class iCaRLExtended(iCaRL):

    def __init__(self, loss, resnet_type="32", num_classes=10, k=2000, classifier='SVC', **classifier_params):
        super(iCaRLExtended, self).__init__(loss, resnet_type, num_classes, k)

        self.classifier_model, self.require_numpy, self.require_train = \
            self.get_classifier_config(classifier, classifier_params)

    def classify(self, batch_images):
        assert self.exemplars_means is not None
        assert self.exemplars_means.shape[0] == self.n_classes

        self.features_extractor.train(False)

        features = self.features_extractor(batch_images)
        features = l2_normalize(features, dim=1)

        if self.require_numpy:
            X = []
            for feature in features:
                X.append(feature.squeeze().cpu().detach().numpy())
            features = X

        preds = self.classifier_model.predict(features)
        return torch.tensor(preds)

    def after_task(self, train_loader, targets):
        super().after_task(train_loader, targets)

        if self.require_train:
            if type(self.classifier_model) is CosineSimilarityClassifier:
                self.classifier_model.fit(self.exemplars_means)
            else:
                X, y = [], []
                for class_idx in range(len(self.exemplars)):
                    for exemplar in self.exemplars[class_idx]:
                        # The features extractor expect a 4-dimensional tensor (for batchs),
                        # therefore we add a extra dimension with unsqueeze(0).
                        exemplar = exemplar.to(Config.DEVICE).unsqueeze(0)
                        # Then the feature is in batch, it needs to remove on dimension with squeeze.
                        feature = self.features_extractor(exemplar).squeeze()
                        X.append(l2_normalize(feature).cpu().detach().numpy())
                    y.extend([class_idx] * len(self.exemplars[class_idx]))

                self.classifier_model.fit(X, y)

    @staticmethod
    def get_classifier_config(classifier, classifier_params):
        # Returns the classifier, if it requires numpy detach features and if it requires a training
        if classifier == 'KNN':
            return KNeighborsClassifier(**classifier_params), True, True
        if classifier == 'SVC':
            return SVC(**classifier_params), True, True
        if classifier == 'COS':
            return CosineSimilarityClassifier(**classifier_params), False, True
        else:
            raise NotImplementedError('Classifier not implemented.')


class CosineSimilarityClassifier:

    def __init__(self, *args, **kwargs):
        self.exemplars_means = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def fit(self, exemplar_means):
        self.exemplars_means = exemplar_means

    def predict(self, features):
        pred_labels = []

        for feature in features:
            # Comparing each exemplar mean with the feature, the feature need another dimension
            # for doing that.
            cos_sim = self.cos(self.exemplars_means, feature.unsqueeze(0))
            pred_labels.append(cos_sim.argmax().item())

        return torch.from_numpy(np.array(pred_labels))
