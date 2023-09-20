'''
Functions in this file take models in the src/models/ folder and optimise their parameters using
preprocessed training data from the data/derived/ folder (e.g. output from feature-design scripts)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sklearn
import os
import pandas as pd

from tqdm.auto import tqdm
from src.models.example_dnn_model import ExampleDNN


def train_example_model(num_epochs=20, lr=1e-3, save_name):
    mod = ExampleDNN(144)
    # ...

    # save the model state to file to avoid retraining
    torch.save(mod.state_dict(), os.path.join("model-checkpoints", save_name))



if __name__ == '__main__':

    train_example_model()
