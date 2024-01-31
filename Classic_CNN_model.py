#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:15:01 2022

@author: liupeilin
"""
import torch
from torch import nn
from torch.nn import functional as F

class AlexNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.BatchNorm = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.BatchNorm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGGNet(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Inception(nn.Module):
    
    def __init__(self, in_channels, channel1, channel2, channel3, channel4):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, channel1, kernel_size=1)
        self.branch2_1 = nn.Conv2d(in_channels, channel2[0], kernel_size=1)
        self.branch2_2 = nn.Conv2d(channel2[0], channel2[1], kernel_size=3, padding=1)
        self.branch3_1 = nn.Conv2d(in_channels, channel3[0], kernel_size=1)
        self.branch3_2 = nn.Conv2d(channel3[0], channel3[1], kernel_size=5, padding=2)
        self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = nn.Conv2d(in_channels, channel4, kernel_size=1)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out1 = F.relu(out1)
        out2 = self.branch2_1(x)
        out2 = F.relu(out2)
        out2 = self.branch2_2(out2)
        out2 = F.relu(out2)
        out3 = self.branch3_1(x)
        out3 = F.relu(out3)
        out3 = self.branch3_2(out3)
        out3 = F.relu(out3)
        out4 = self.branch4_1(x)
        out4 = self.branch4_2(out4)
        out4 = F.relu(out4)
        concat = torch.cat((out1, out2, out3, out4), dim=1)
        return concat
    
class GoogLeNet(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.inception3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.inception4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.inception4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.inception5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes, bias=True)
        
    def forward(self, x):
        feature = self.conv1(x)
        feature = self.conv2(feature)
        feature = self.inception3_1(feature)
        feature = self.inception3_2(feature)
        feature = self.max_pool3(feature)
        feature = self.inception4_1(feature)
        feature = self.inception4_2(feature)
        feature = self.inception4_3(feature)
        feature = self.inception4_4(feature)
        feature = self.inception4_5(feature)
        feature = self.max_pool(feature)
        feature = self.inception5_1(feature)
        feature = self.inception5_2(feature)
        feature = self.avg_pool(feature)
        feature = self.dropout(feature)
        out = self.fc(feature.view(x.shape[0], -1))
        return out
