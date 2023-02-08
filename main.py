import time

from tqdm import tqdm
import click

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock

from models import CNNModel, ClassifierModel, AttentionModel, MILModel
from datasets import MNISTDataset, MNISTMILDataset


@click.group()
def cli():
    pass

@cli.command()
@click.option('--mil', 'use_mil', is_flag=True)
def train(use_mil):
    matplotlib.use('Agg')
    EPOCH = 500

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset = MNISTMILDataset(train=True)
    cnn_model = CNNModel(num_classes=10)
    cls_model = ClassifierModel(num_features=cnn_model.num_features, num_classes=10)
    attentio_model = AttentionModel(num_features=cnn_model.num_features)

    cnn_optimizer = optim.RAdam(cnn_model.parameters(), lr=0.01)
    cls_optimizer = optim.RAdam(cls_model.parameters(), lr=0.01)
    attention_optimizer = optim.RAdam(attentio_model.parameters(), lr=0.01)

    cross_entropy = nn.CrossEntropyLoss()

    t_epoch = tqdm(range(EPOCH))
    epoch_base_losses = []
    epoch_accs = []
    epoch_mil_losses = []
    epoch_mil_acc = []
    for epoch in t_epoch:
        t_batch = tqdm(range(len(train_dataset)), leave=False)
        base_losses = []
        mil_losses = []
        correction = []
        mil_correction = []
        for i in t_batch:
            x, gt = train_dataset[i]
            x = x.to(device)
            gt = gt.to(device)

            cnn_optimizer.zero_grad()
            cls_optimizer.zero_grad()
            attention_optimizer.zero_grad()

            features = cnn_model(x)
            preds = cls_model(features)
            attentions = attentio_model(features)

            pred = torch.softmax((preds * attentions[:, None]).sum(dim=0), dim=-1)
            loss_bag = cross_entropy(pred, gt)
            print('loss_bag', loss_bag)

            loss_instances = cross_entropy(preds, gt.repeat(preds.shape[0]))

            print('loss_instances', loss_instances)

            attentions = attentio_model(features)
            preds = cls_model(features)

            base_loss = criterion(preds, gts)
            p = torch.argmax(preds, dim=1)

            has_zero = torch.any(gts == 0).to(torch.float32)
            attention = mil(preds)
            pred = torch.sigmoid((preds * attention[:, None]).sum())
            mil_loss = mil_criterion(pred, has_zero)
            loss = base_loss + mil_loss * 0.3
            # p = (preds > 0.5).flatten()
            mil_correction += [(pred > 0.5) == (has_zero > 0.5)]
            mil_losses += [mil_loss.item()]

            c = (gts == p).tolist()
            acc = np.sum(c) / len(c)
            correction += c
            base_losses += [base_loss.item()]
            loss.backward()
            optimizer.step()
            m = f'[{epoch}/{EPOCH}] loss: {loss:.3f} acc:{acc:.3f}'
            if use_mil:
                m += f'mil_loss:{mil_loss:.3f}'
            t_batch.set_description(m)
            t_batch.refresh()

        acc = np.sum(correction) / len(correction)
        epoch_base_losses.append(np.mean(base_losses))
        epoch_accs.append(acc)
        m = f'epoch loss: {epoch_base_losses[-1]:.3f} acc:{acc:.3f}'
        if use_mil:
            epoch_mil_losses.append(np.mean(mil_losses))
            mil_acc = np.sum(mil_correction) / len(mil_correction)
            epoch_mil_acc.append(mil_acc)
            m += f' mil:{mil_acc:.3f}'
            m += ' a:' + ','.join([f'{f:.3f}' for f in attention.flatten().tolist()])
        t_epoch.set_description(m)
        t_epoch.refresh()

        plt.plot(epoch_base_losses, label='loss')
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
