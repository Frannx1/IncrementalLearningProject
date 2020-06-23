import numpy as np
import torch
from torch import nn

from config import Config
from models import iCaRL
from models.utils import l2_normalize

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from models.utils.utilities import replace_row


class iCaRLExtended(iCaRL):

    def __init__(self, loss, resnet_type="32", num_classes=10, k=2000, classifier='SVC', **classifier_params):
        super(iCaRLExtended, self).__init__(loss, resnet_type, num_classes, k)

        self.classifier_model, self.require_normalize, self.require_train = \
            self.get_classifier_config(classifier, classifier_params)

        self.nn_exemplars_means = None     # Not normalized exemplars means

    def classify(self, batch_images):
        assert self.exemplars_means is not None
        assert self.exemplars_means.shape[0] == self.n_classes

        self.features_extractor.train(False)

        features = self.features_extractor(batch_images)

        if self.require_normalize:
            features = l2_normalize(features, dim=1)
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
                self.classifier_model.fit(self.nn_exemplars_means)
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
        # Returns the classifier, if it requires normalized features and if it requires a training
        if classifier == 'KNN':
            return KNeighborsClassifier(**classifier_params), True, True
        if classifier == 'SVC':
            return SVC(**classifier_params), True, True
        if classifier == 'COS':
            return CosineSimilarityClassifier(**classifier_params), False, True
        else:
            raise NotImplementedError('Classifier not implemented.')

    def _extract_features_and_mean(self, dataloader):
        if type(self.classifier_model) is CosineSimilarityClassifier:
            # For Cosine Similarity it is required the means without normalization
            nn_features = []
            with torch.no_grad():
                for images, class_idx in dataloader:
                    images = images.to(Config.DEVICE)
                    nn_features.append(self.features_extractor(images))

                nn_features = torch.cat(nn_features)
                features = l2_normalize(nn_features, dim=1)
                nn_mean = nn_features.mean(dim=0)
                mean = features.mean(dim=0)

            self._expand_nn_exemplars_means(class_idx[0].item(), nn_mean)
            return features, l2_normalize(mean)
        else:
            return super()._extract_features_and_mean(dataloader)

    def _expand_nn_exemplars_means(self, class_idx, nn_mean):
        if self.nn_exemplars_means is None:
            assert class_idx == 0
            self.nn_exemplars_means = nn_mean[None, ...]
        else:
            # Checking if the new class follows the previous ones
            if self.nn_exemplars_means.shape[0] == class_idx:
                self.nn_exemplars_means = torch.cat((self.nn_exemplars_means, nn_mean[None, ...]))
            elif self.nn_exemplars_means.shape[0] > class_idx:
                self.exemplars_means = replace_row(self.nn_exemplars_means, nn_mean, class_idx)
            else:
                raise ValueError('Adding a not normalized exemplar mean in an incorrect index.')


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
