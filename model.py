from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,256,(3,3)),
            nn.ReLU(),
            nn.Conv2d(256,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(224-6)*(224-6), 64),
            nn.Linear(64,64),
            nn.Linear(64,2)
            )
    def forward(self,x):
        return self.model(x)
    
