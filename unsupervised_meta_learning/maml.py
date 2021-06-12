# AUTOGENERATED! DO NOT EDIT! File to edit: 02_maml_pl.ipynb (unless otherwise specified).

__all__ = ['logger', 'ConvolutionalNeuralNetwork', 'get_accuracy', 'MAML', 'UMTRA']

# Cell
#export
import logging
import warnings

import higher
import kornia as K
import wandb
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from .pl_dataloaders import OmniglotDataModule

# Cell
logger = logging.getLogger(__name__)

# Cell
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            self.conv3x3(in_channels, hidden_size),
            self.conv3x3(hidden_size, hidden_size),
            self.conv3x3(hidden_size, hidden_size),
            self.conv3x3(hidden_size, hidden_size),
        )

        self.classifier = nn.Linear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits

    def conv3x3(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
            nn.BatchNorm2d(out_channels, momentum=1.0, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

# Cell
def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

# Cell
class MAML(pl.LightningModule):
    def __init__(self, model, inner_steps=1):
        super().__init__()
        self.model = model
        self.accuracy = get_accuracy
        self.automatic_optimization = False
        self.inner_steps = inner_steps

    def forward(self, x):
        return self.model(x)

    def inner_loop(self, fmodel, diffopt, train_input, train_target):
        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)

        return inner_loss.item()

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, optimizer_idx=None):
        meta_optimizer, inner_optimizer = self.optimizers()
        meta_optimizer = meta_optimizer.optimizer
        inner_optimizer = inner_optimizer.optimizer

        train_inputs, train_targets = batch['train']
        test_inputs, test_targets = batch['test']

        batch_size = train_inputs.shape[0]
        outer_loss = torch.tensor(0., device=self.device)
        acc = torch.tensor(0., device=self.device)
        self.model.zero_grad()

        for task_idx, (train_input, train_target, test_input, test_target) in enumerate(
            zip(train_inputs, train_targets, test_inputs, test_targets)
        ):
#             inner_optimizer.zero_grad()
            with higher.innerloop_ctx(self.model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
#                 train_logit = fmodel(train_input)
#                 inner_loss = F.cross_entropy(train_logit, train_target)

#                 diffopt.step(inner_loss)
                for step in range(self.inner_steps):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                test_logit = fmodel(test_input)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    preds = test_logit.softmax(dim=-1)
                    acc += self.accuracy(test_logit, test_target)


#                     self.print(self.accuracy(test_logit, test_target))

        outer_loss.div_(batch_size)
        acc.div_(batch_size)
        self.log_dict({
                    'outer_loss': outer_loss,
                    'accuracy': acc
                }, prog_bar=True)

        meta_optimizer.zero_grad()
#         outer_loss.backward()
        self.manual_backward(outer_loss, meta_optimizer)
        meta_optimizer.step()
        return outer_loss, acc


    def training_step(self, batch, batch_idx, optimizer_idx):
        train_loss, acc = self.meta_learn(batch, batch_idx, optimizer_idx)

        self.log_dict({
            'train_loss': train_loss.item(),
            'train_accuracy': acc.item()
        }, prog_bar=True)

        return train_loss.item()

    def validation_step(self, batch, batch_idx):
        val_loss, val_acc = self.meta_learn(batch, batch_idx)

        self.log_dict({
            'val_loss': val_loss.item(),
            'val_accuracy': val_acc.item()
        })
        return val_loss.item()

    def test_step(self, batch, batch_idx):
        test_loss, test_acc = self.meta_learn(batch, batch_idx)
        self.log_dict({
            'test_loss': test_loss.item(),
            'test_accuracy': test_acc.item()
        })
        return test_loss.item()


    def configure_optimizers(self):
        meta_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        inner_optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)

        return [meta_optimizer, inner_optimizer]



# Cell
class UMTRA(pl.LightningModule):
    def __init__(self, model, augmentation, inner_steps):
        super().__init__()
        self.model = model
        self.accuracy = get_accuracy
        self.augmentation = augmentation
        self.inner_steps = inner_steps
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def inner_loop(self, fmodel, diffopt, train_input, train_target):
        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)

        return inner_loss.item()

    def training_step(self, batch, batch_idx, optimizer_idx):
        meta_optimizer, inner_optimizer = self.optimizers(use_pl_optimizer=False)
        train_inputs, train_targets = batch['train']
        test_inputs, test_targets = batch['test']

        batch_size = train_inputs.shape[0]
        outer_loss = torch.tensor(0., device=self.device)
        acc = torch.tensor(0., device=self.device)
        self.model.zero_grad()

        for task_idx, (train_input, train_target, test_input, test_target) in enumerate(
            zip(train_inputs, train_targets, test_inputs, test_targets)
        ):
            val_input = self.augmentation(train_input).to(self.device)
            val_target = deepcopy(train_target).to(self.device)
            with higher.innerloop_ctx(self.model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                for step in range(self.inner_steps):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                val_logits = fmodel(val_input)
                outer_loss += F.cross_entropy(val_logits, val_target)

                with torch.no_grad():
                    test_logits = fmodel(test_input)
                    acc += self.accuracy(test_logits, test_target)

        outer_loss.div_(batch_size)
        acc.div_(batch_size)
        self.log_dict({
                    'outer_loss': outer_loss,
                    'accuracy': acc
                }, prog_bar=True)

        meta_optimizer.zero_grad()
#         outer_loss.backward()

        self.manual_backward(outer_loss, meta_optimizer)
        meta_optimizer.step()

        return outer_loss

    def configure_optimizers(self):
        meta_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        inner_optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)

        return [meta_optimizer, inner_optimizer]