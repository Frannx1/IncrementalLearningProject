from abc import abstractmethod

from torch.utils.data import Dataset


class IncrementalDataset(Dataset):

    @abstractmethod
    def append(self, data, targets):
        """Append dataset with images and labels
        Args:
            data: Tensor of shape (N, C, H, W)
            targets: list of labels
        """
        pass

    @abstractmethod
    def get_data(self, index):
        """Returns the data image without any transformation
        Args:
            index: the position in the dataset of the image
        """
        pass
