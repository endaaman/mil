import time

from tqdm import tqdm
import click

import pandas as pd
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

    cross_entropy = nn.CrossEntropyLoss(reduction='none')

    t_epoch = tqdm(range(EPOCH))
    metrics = []
    cols = [
        'loss_instances',
        'loss_overall',
        'acc_instances',
        'acc_overall',
    ]

    df = pd.DataFrame(metrics, columns=cols)
    for epoch in t_epoch:
        t_batch = tqdm(range(len(train_dataset)), leave=False)
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
            loss_bag = cross_entropy(pred, gt).mean()
            loss_instances = cross_entropy(preds, gt.repeat(preds.shape[0]))

            beta = -attentions + torch.max(attentions)

            loss_regulation = -(loss_instances * beta).mean()
            loss_overall = loss_bag - loss_regulation

            cls_optimizer.zero_grad()
            cnn_optimizer.zero_grad()
            loss_overall.backward()
            cls_optimizer.step()
            cnn_optimizer.step()

            attention_optimizer.zero_grad()
            loss_instances = cross_entropy(preds, gt.repeat(preds.shape[0]))
            loss_instances.mean().backward(retain_graph=True)
            attention_optimizer.step()

            m = f'[{epoch}/{EPOCH}] overall:{loss_overall:.3f} instance:{loss_instances:.3f}'

            metrics.append([
                loss_instances.item(),
                loss_overall.item(),
                pred.argmax() == gt,
            ])

            t_batch.set_description(m)
            t_batch.refresh()

        df_current = pd.DataFrame(metrics, columns=cols).mean()
        # [sum(v, []) for v in zip(*l)]

        metrics_messages = []
        for k, v in df.mean().items():
            metrics_messages.append(f'{k}:{v:.3f}')
        metrics_message = ' '.join(metrics_messages)
        t_epoch.set_description(metrics_message)
        t_epoch.refresh()

        for k, v in df.mean().items():
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
