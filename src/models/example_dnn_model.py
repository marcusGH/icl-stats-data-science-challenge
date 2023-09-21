import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sklearn
import os
import pandas as pd


class ExampleDNN(nn.Module):

    def __init__(self, num_features):
        self.feed_forward = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.feed_forward(x)
