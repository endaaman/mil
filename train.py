import time
from tqdm import tqdm

import click
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock

from endaaman.torch import Trainer, TorchCommander, pil_to_tensor, tensor_to_pil
from endaaman.metrics import MultiAccuracy, AccuracyByChannel, MetricsFn
from endaaman.functional import multi_accuracy


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

class ToyResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.base.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.base(x)
        return x


class MILModel(nn.Module):
    def __init__(self, model, num_classes, params_size=100):
        super().__init__()
        self.model = model
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
        # x: (i, ) u: (p, i, ) v: (p, i, )
        xu = torch.tanh(torch.matmul(self.u, x))
        xv = torch.sigmoid(torch.matmul(self.v, x))
        # xu: (p, ) xu: (p, )
        x = xu * xv
        # x: (p, ) w: (p, i, )
        attention = torch.matmul(x, self.w)
        return attention

    def forward_attention(self, preds):
        '''
        Args:
            preds (Tensor): (B, C) batched logits
        Returns:
            Tensor: attention P-values (B, ) [0, 1]
        '''
        instances = []
        for pred in preds:
            instances.append(self.compute_attention_scores(pred))
        attention = torch.stack(instances)
        attention = torch.softmax(attention.flatten(), dim=0)
        return attention


    def forward(self, x):
        preds = self.model(x)
        attention = self.forward_attention(preds)
        # preds: (B, C) attention: (B, )
        total_pred = (preds * attention[:, None]).sum()
        return preds, total_pred



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



class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        assert len(kwargs.items()) == 0
        self.base_criterion = nn.CrossEntropyLoss()
        self.mil_criterion = nn.BCELoss()

        cnn = ToyModel(num_classes=10)
        model = MILModel(model=cnn, num_classes=10)
        return model

    def eval(self, x, gts):
        preds, total_pred = self.model(x.to(self.device))
        preds = preds.cpu()
        total_pred = total_pred.cpu()
        base_loss = self.base_criterion(preds, gts)
        total_gt = torch.any(gts == 0).to(torch.float32)
        mil_loss = self.mil_criterion(torch.sigmoid(total_pred), total_gt)
        loss = base_loss + mil_loss * 0.3
        # loss = base_loss
        return loss, (preds.detach(), total_pred[None].detach())

    def get_metrics(self):
        return {
            'batch': {
                'acc': MultiAccuracy(selector=lambda p, g: (p[0], g)),
                # 'mil_acc': BinaryAccuracy(selector=lambda p, g: (p[1], g)),
            },
            'epoch': { },
        }


class CMD(TorchCommander):
    def arg_common(self, parser):
        pass

    def arg_start(self, parser):
        parser.add_argument('--use-mil', '-m', action='store_true')

    def run_start(self):
        loaders = [
            self.as_loader(MNISTDataset(train=t), shuffle=False) for t in [True, False]
        ]

        trainer = self.create_trainer(
            T=MyTrainer,
            model_name='toy',
            loaders=loaders,
        )
        trainer.start(self.a.epoch)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 30,
        'lr': 0.01,
        'save_period': -1,
        'batch_size': 9,
    })
    cmd.run()
