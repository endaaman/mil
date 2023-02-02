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
        # self.base = /esNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        # self.base.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = self.base(x)
        # return torch.sigmoid(x)
        x = self.convs(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




class MILAttention(nn.Module):
    def __init__(self, num_classes, params_size=100):
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
        # x: (i, )
        # u: (p, i, )
        # v: (p, i, )
        xu = torch.tanh(torch.matmul(self.u, x))
        xv = torch.sigmoid(torch.matmul(self.v, x))

        # xu: (p, )
        # xu: (p, )
        x = xu * xv

        # x: (p, )
        # w: (p, i, )
        attention = torch.matmul(x, self.w)
        return attention

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

        # if self.num_classes > 1:
        #     x = torch.softmax(x, dim=-1)
        # else:
        #     x = torch.sigmoid(x)
        attention = torch.stack(instances)
        attention = torch.softmax(attention.squeeze(), dim=0)
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
        x = self.base.data[idx]
        y = self.base.targets[idx]

        # x = make_grid(xs.view(9, 1, 28, 28), nrow=3, padding=0)[0]
        # x = x.float() / 255
        x = x[:, None, :, :].to(torch.float32)
        x = x / 255
        return x, y

    def __len__(self):
        return 100



@cli.command()
@click.option('--mil', 'use_mil', is_flag=True)
def train(use_mil):
    EPOCH = 100

    train_dataset = TileDataset(train=True, y_is_including_zero=use_mil)

    model = ToyModel(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    if use_mil:
        mil = MILAttention(num_classes=10, params_size=100)
        mil_criterion = nn.BCELoss()
        params = list(model.parameters()) + list(mil.parameters())
    else:
        params = model.parameters()

    optimizer = optim.RAdam(params, lr=0.01)

    t_epoch = tqdm(range(EPOCH))
    epoch_losses = []
    epoch_accs = []
    epoch_mil_losses = []
    epoch_mil_acc = []
    for epoch in t_epoch:
        t_batch = tqdm(range(len(train_dataset)), leave=False)
        losses = []
        mil_losses = []
        correction = []
        mil_correction = []
        for i in t_batch:
            x, gts = train_dataset[i]
            optimizer.zero_grad()
            preds = model(x)
            base_loss = criterion(preds, gts)
            p = torch.argmax(preds, dim=1)
            if use_mil:
                has_zero = torch.any(gts == 0).to(torch.float32)
                attention = mil(preds)
                pred = torch.sigmoid((preds * attention[:, None]).sum())
                mil_loss = mil_criterion(pred, has_zero)
                loss = base_loss + mil_loss
                loss = base_loss
                # p = (preds > 0.5).flatten()
                mil_correction += [(pred > 0.5) == (has_zero > 0.5)]
                mil_losses += [mil_loss.item()]
            else:
                loss = base_loss
            c = (gts == p).tolist()
            acc = np.sum(c) / len(c)
            correction += c
            losses += [base_loss.item()]
            loss.backward()
            optimizer.step()
            m = f'loss: {loss:.3f} acc:{acc:.3f}'
            if use_mil:
                m += f'mil_loss:{mil_loss:.3f}'
            t_batch.set_description(m)
            t_batch.refresh()

        acc = np.sum(correction) / len(correction)
        epoch_losses.append(np.mean(losses))
        epoch_accs.append(acc)
        m = f'epoch loss: {epoch_losses[-1]:.3f} acc:{acc:.3f}'
        if use_mil:
            epoch_mil_losses.append(np.mean(mil_losses))
            mil_acc = np.sum(mil_correction) / len(mil_correction)
            epoch_mil_acc.append(mil_acc)
            m += f' mil:{mil_acc:.3f}'
            m += ' a:' + ','.join([f'{f:.3f}' for f in attention.flatten().tolist()])
        t_epoch.set_description(m)
        t_epoch.refresh()

        plt.plot(epoch_losses, label='loss')
        plt.plot(epoch_accs, label='acc')
        if use_mil:
            plt.plot(epoch_mil_losses, label='mil loss')
            plt.plot(epoch_mil_acc, label='mil acc')
        plt.legend()
        plt.grid()
        plt.savefig('out/mil.png' if use_mil else 'out/base.png')
        plt.close()


    # model = ToyModel()
    # t = torch.ones(2, 1, 28, 28)

    # print(model(t).sum())
    # print(model(t).shape)

@cli.command()
def MIL_test():
    instance_count = 3
    num_classes = 2

    preds = torch.randn(instance_count, num_classes)
    mil = MILAttention(num_classes=num_classes)
    attention = mil(preds)
    print(attention)

@cli.command()
def model_test():
    model = ToyModel(num_classes=10)
    x = torch.randn(3, 1, 28, 28)
    y = model(x)
    print(y.shape)
    print(type(model.parameters()))

if __name__ == '__main__':
    cli()
