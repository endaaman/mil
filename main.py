import time
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock

import click

@click.group()
def cli():
    pass


class ToyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.base.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.base(x)
        # return torch.sigmoid(x)
        return x


class MILAttention(nn.Module):
    def __init__(self, num_classes, params_size=10):
        super().__init__()
        self.num_classes = num_classes
        self.params_size = params_size
        self.u = nn.Parameter(torch.randn(params_size, num_classes))
        self.v = nn.Parameter(torch.randn(params_size, num_classes))
        self.w = nn.Parameter(torch.randn(params_size, 1))

    def compute_attention_scores(self, x):
        '''
        Args:
            x (Tensor): (C, ) logits by instance
        Returns:
            Tensor: (1, )
        '''
        #  w^T*(tanh(u*h_k^T)*sigmoid(v*h_k^T))
        xu = torch.tanh(torch.matmul(self.u, x))
        xv = torch.sigmoid(torch.matmul(self.v, x))
        x = xu * xv
        x = torch.matmul(x, self.w)
        return x

    def forward(self, preds):
        '''
        Args:
            preds (Tensor): (B, C) batched logits
        Returns:
            Tensor: attention P-values (B, ) [0, 1]
        '''
        instances = []
        for pred in preds:
            instances.append(self.compute_attention_scores(pred))
        attention = torch.softmax(torch.stack(instances).squeeze(), dim=-1)
        return attention


class TileDataset(Dataset):
    def __init__(self, train=True, y_is_including_zero=True):
        self.y_is_including_zero = y_is_including_zero
        self.base = datasets.MNIST(
            './data/mnist',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()
            ]),
        )

    def __getitem__(self, i):
        idx = torch.randint(len(self.base.data), [9])
        xs = self.base.data[idx]
        ys = self.base.targets[idx]

        # x = make_grid(xs.view(9, 1, 28, 28), nrow=3, padding=0)[0]
        # x = x.float() / 255
        xs = xs[:, None, :, :]
        xs = (xs / 255) * 2 - 1.0

        if self.y_is_including_zero:
            # include "0"
            y = torch.any(ys == 0)
            y = y.to(torch.float32)[None]
        else:
            # y = (ys == 0).to(torch.float32)
            y = ys
        return xs, y

    def __len__(self):
        return 100



@cli.command()
def train():
    EPOCH = 10000

    model = ToyModel(num_classes=10)
    attention = MILAttention(num_classes=1, params_size=100)


    as_tile = False
    if as_tile:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    train_dataset = TileDataset(train=True, y_is_including_zero=as_tile)


    optimizer = optim.RAdam(model.parameters(), lr=0.001)

    t_epoch = tqdm(range(EPOCH))
    epoch_losses = []
    epoch_accs = []
    for epoch in t_epoch:
        t_batch = tqdm(range(len(train_dataset)), leave=False)
        correction = []
        losses = []
        for i in t_batch:
            xs, y = train_dataset[i]

            optimizer.zero_grad()
            preds = model(xs)
            if as_tile:
                preds = attention(preds)
                preds = torch.sigmoid(preds)
                loss = criterion(preds, y)
            else:
                # preds = torch.sigmoid(preds)
                # preds = torch.softmax(preds, dim=-1)
                # print(preds.shape)
                # print(y.shape)
                # print(preds)
                # print(y)
                loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            c = (y == torch.argmax(preds[0])).tolist()
            correction += c
            acc = np.sum(c) / len(c)
            losses.append(loss.item())
            t_batch.set_description(f'loss: {loss:.3f} acc:{acc:.3f}')
            t_batch.refresh()


        acc = np.sum(correction) / len(correction)
        epoch_losses.append(np.mean(losses))
        epoch_accs.append(acc)
        t_epoch.set_description(f'epoch loss: {epoch_losses[-1]:.3f} acc:{acc:.3f}')
        t_epoch.refresh()

        plt.plot(epoch_losses)
        plt.plot(epoch_accs)
        plt.savefig('out/loss.png')
        plt.close()


    # model = ToyModel()
    # t = torch.ones(2, 1, 28, 28)

    # print(model(t).sum())
    # print(model(t).shape)

@cli.command()
def MIL_test():
    instance_count = 3
    num_classes = 10

    preds = torch.randn(instance_count, num_classes)
    mil = MILAttention(num_classes=num_classes)
    attention = mil(preds)
    print(attention)

if __name__ == '__main__':
    cli()
