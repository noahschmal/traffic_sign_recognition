import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes, pretrained=False):
    model = models.resnet50(pretrained=False)

    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

class_names = ['keepRight', 'merge', 'pedestrianCrossing', 'signalAhead', 
               'speedLimit25', 'speedLimit35', 'stop', 'yield', 'yieldAhead']

model = create_model(num_classes=9)
 
