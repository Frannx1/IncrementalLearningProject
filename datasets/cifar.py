from PIL import Image
from torchvision.datasets import CIFAR100

import numpy as np


class iCIFAR100(CIFAR100):
    """ A Dataset containing specified classes from the CIFAR100 dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        classes (iterable, optional): The iterator containing the classes indexes wanted
            of the CIFAR100 dataset. By default all the classes are loaded.
    """

    def __init__(self,
                 root='./data',
                 train=True,
                 download=True,
                 transform=None,
                 target_transform=None,
                 classes=range(100)):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        download=download,
                                        transform=transform,
                                        target_transform=target_transform)

        # Select subset of classes
        data = []
        targets = []

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                data.append(self.data[i])
                targets.append(self.targets[i])

        self.data = np.array(data)
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_class_images(self, label):
        return self.data[np.array(self.targets) == label]


class iCIFARSplit:
    """ An iterator that contains the CIFAR100 dataset split in a given number of groups,
        each split contains a continuous unique list of classes. For each iteration it
        returns a train dataset and a test dataset. The train dataset only contains the
        classes of that iteration, while the test dataset contains all the classes seen
        until that iteration.

    Args:
        total_groups (int): The total number of groups to split the dataset.
        train_transform (callable, optional): A function/transform that takes in an training
            PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        test_transform (callable, optional): Similar to train_transform parameter, but
            this one is applied to the test dataset.
        total_classes (int): The number of total classes of the CIFAR100 dataset expected
            to have in the iterator.
    """

    def __init__(self, total_groups=10, train_transform=None, test_transform=None, total_classes=100):
        if total_classes % total_groups != 0:
            raise ValueError('Only a total number of groups that can divide the number of total classes is accepted.')
        self.class_per_group = total_classes / total_groups

        self.total_groups = total_groups
        self.train_groups = []
        self.test_groups = []
        self.train_groups_classes = {}
        self.current_iter = -1

        for group_idx in range(total_groups):
            first_class = int(group_idx * self.class_per_group)
            last_class = int((group_idx + 1) * self.class_per_group)
            range_classes = range(first_class, last_class)
            download = group_idx == 0

            self.train_groups_classes[group_idx] = range_classes
            self.train_groups.append(iCIFAR100(transform=train_transform, classes=range_classes,
                                               download=download))
            self.test_groups.append(iCIFAR100(transform=test_transform, train=False,
                                              classes=range(last_class), download=False))

    def get_train_group(self, i):
        return self.train_groups[i]

    def get_test_group(self, i):
        return self.test_groups[i]

    def get_total_groups(self):
        return self.total_groups

    def get_train_groups_classes(self):
        return self.train_groups_classes

    def __len__(self):
        return self.total_groups

    def __iter__(self):
        self.current_iter = -1
        return self

    def __next__(self):
        self.current_iter += 1
        if self.current_iter < self.total_groups:
            return self.train_groups[self.current_iter], self.test_groups[self.current_iter]
        raise StopIteration

