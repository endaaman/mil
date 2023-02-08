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

from models import CNNModel, ClassifierModel, AttentionModel, MILModel
from datasets import MNISTDataset

class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        assert len(kwargs.items()) == 0
        self.base_criterion = nn.CrossEntropyLoss()
        self.mil_criterion = nn.BCELoss()

        cnn_model = CNNModel(num_classes=10)
        cls_model = ClassifierModel(num_features=cnn_model.num_features, num_classes=10)
        attentio_model = AttentionModel(num_features=cnn_model.num_features)
        model = MILModel(cnn_model, cls_model, attentio_model)
        return model

    def eval(self, inputs, gts):
        inputs = inputs.to(self.device)
        features = self.model.cnn_model(inputs)
        attentions = self.model.attentio_model(features)
        preds = self.model.cls_model(features)

        base_loss = self.base_criterion(preds, gts)
        total_gt = torch.any(gts == 0).to(torch.float32)
        mil_loss = self.mil_criterion(torch.sigmoid(total_pred), total_gt)
        loss = base_loss + mil_loss * 0.3


        # self.optimizer.zero_grad()
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
