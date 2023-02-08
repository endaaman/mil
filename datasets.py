import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.utils import make_grid
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, train=True, y_is_including_zero=True):
        self.train = train
        self.base = datasets.MNIST(
            './data/mnist',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()
            ]),
        )

    def __getitem__(self, i):
        # idx = torch.randint(len(self.base.data), [9])
        i = np.random.randint(len(self.base.data))
        x = self.base.data[i]
        y = self.base.targets[i]

        # x = make_grid(xs.view(9, 1, 28, 28), nrow=3, padding=0)[0]
        # x = x.float() / 255
        x = x[None, :, :].to(torch.float32) / 255
        return x, y

    def __len__(self):
        return 10000 if self.train else 1000


class MNISTMILDataset(Dataset):
    def __init__(self, train=True, y_is_including_zero=True):
        self.train = train
        self.base = datasets.MNIST(
            './data/mnist',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()
            ]),
        )

    def __getitem__(self, i):
        main_count = 3
        noise_count = 3

        target_label = random.randint(0, 9)
        target_indexes = np.where(self.base.targets == target_label)[0]

        main_indexes = np.random.choice(target_indexes, main_count)
        main_x = self.base.data[main_indexes]

        noise_indexes = torch.randint(0, len(self.base.data), [noise_count])
        noise_x = self.base.data[noise_indexes]

        x = torch.cat([main_x, noise_x])
        x = x[torch.randperm(x.shape[0])] # shuffle
        x = x[:, None, :, :].to(torch.float32) / 255
        y = torch.tensor(target_label)
        return x, y

    def __len__(self):
        return 10000 if self.train else 1000
