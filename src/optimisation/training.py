import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import numpy as np
import sklearn
import pandas as pd
import os
# import cupy
# import cudf
import sklearn

from datetime import datetime
from tqdm.auto import tqdm

class EarlyStopper:
    """
    References:
      - https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, loss_fn, training_loader, optimizer, epoch_number, dev=torch.device('cuda')):
    running_loss = 0.
    running_metric = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(dev), labels.to(dev)

        # sanity check
        if not torch.all(torch.isfinite(inputs)):
            msg = f"Encountered {len(inputs[~torch.isfinite(inputs)])} non-finite input values at iteration {i} during training"
            raise ValueError(msg)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_metric += amex_metric_mod(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())

    last_loss = running_loss / (i + 1) # loss per batch
    last_metric = running_metric / (i + 1) # metric per batch
    # print('  batch {} loss: {} metric: {}'.format(i + 1, last_loss, last_metric))
    # tb_x = epoch_index * len(training_loader) + i + 1
    # tb_writer.add_scalar('Loss/train', last_loss, tb_x)

    return last_loss, last_metric
