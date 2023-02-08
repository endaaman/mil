import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock


class CNNModel(nn.Module):
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
        self.num_features = self.fc.in_features

    def forward(self, x):
        # x = self.base(x)
        # return torch.sigmoid(x)
        x = self.convs(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

class ClassifierModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class AttentionModel(nn.Module):
    def __init__(self, num_features, params_count=100):
        super().__init__()
        self.num_features = num_features
        self.u = nn.Parameter(torch.randn(params_count, num_features))
        self.v = nn.Parameter(torch.randn(params_count, num_features))
        self.w = nn.Parameter(torch.randn(params_count, 1))

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
        alpha = torch.matmul(x, self.w)
        return alpha

    def forward(self, features):
        '''
        Args:
            features (Tensor): (B, C) batched features
        Returns:
            Tensor: attention P-values (B, ) [0, 1]
        '''
        attentions = []
        for pred in features:
            attentions.append(self.compute_attention_scores(pred))
        attentions = torch.stack(attentions)
        attentions = torch.softmax(attentions.flatten(), dim=0)
        return attentions


class MILModel(nn.Module):
    def __init__(self, cnn_model, cls_model, attentio_model):
        super().__init__()
        self.cnn_model = cnn_model
        self.cls_model = cls_model
        self.attentio_model = attentio_model


