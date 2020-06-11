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
