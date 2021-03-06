# AUTOGENERATED! DO NOT EDIT! File to edit: 03_protonet_pl.ipynb (unless otherwise specified).

__all__ = ['PrototypicalNetwork', 'CactusPrototypicalModel', 'ProtoModule']

# Cell
#export
import warnings

import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from operator import itemgetter

from .nn_utils import Flatten, get_proto_accuracy, conv3x3
from .pl_dataloaders import OmniglotDataModule

# Cell
class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)

# Cell
class CactusPrototypicalModel(nn.Module):
    def __init__(self, in_channels, hidden_size=64):
        super(CactusPrototypicalModel, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            Flatten()
        )
    def forward(self, inputs):
        return self.encoder(inputs)

# Cell
class ProtoModule(pl.LightningModule):
    def __init__(self, encoder, num_classes, cactus_flag, lr, **kwargs):
        super().__init__()
        self.model = encoder
        self.automatic_optimization = True
        self.num_classes_per_task = num_classes
        self.acccuracy = get_proto_accuracy
        self.cactus_flag = cactus_flag
        self.training_step = self.default_traing_step if not cactus_flag else self.cactus_training_step
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def get_prototypes(self, emb, targets, num_classes):
        """Compute the prototypes (the mean vector of the embedded training/support
        points belonging to its class) for each classes in the task.
        Parameters
        ----------
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the support points. This tensor
            has shape `(batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the support points. This tensor has
            shape `(batch_size, num_examples)`.
        num_classes : int
            Number of classes in the task.
        Returns
        -------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        """

        batch_size, emb_size = emb.size(0), emb.size(-1)

        num_samples = self.get_num_samples(targets, num_classes, dtype=emb.dtype)
        num_samples.unsqueeze_(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        prototypes = emb.new_zeros((batch_size, num_classes, emb_size))
        indices = targets.unsqueeze(-1).expand_as(emb)

        prototypes.scatter_add_(1, indices, emb).div_(num_samples)

        return prototypes

    def get_num_samples(self, targets, num_classes, dtype=None):
        batch_size = targets.size(0)
        with torch.no_grad():
            ones = torch.ones_like(targets, dtype=dtype)
            num_samples = ones.new_zeros((batch_size, num_classes))
            num_samples.scatter_add_(1, targets, ones)
        return num_samples

    def protoypical_loss(self, prototypes, emb, targets, **kwargs):
        """Compute the loss (i.e. negative log-likelihood) for the prototypical
        network, on the test/query points.
        Parameters
        ----------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the query points. This tensor has
            shape `(batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the query points. This tensor has
            shape `(batch_size, num_examples)`.
        Returns
        -------
        loss : `torch.FloatTensor` instance
            The negative log-likelihood on the query points.
        """
        squared_distances = torch.sum(
            (prototypes.unsqueeze(2) - emb.unsqueeze(1)) ** 2, dim = -1
        )
        return F.cross_entropy(-squared_distances, targets, **kwargs)


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def default_traing_step(self, batch, batch_idx):
        train_inputs, train_targets = batch['train']
        train_inputs, train_targets = train_inputs.to(self.device), train_targets.to(self.device)
        test_inputs, test_targets = batch['test']
        test_inputs, test_targets = test_inputs.to(self.device), test_targets.to(self.device)

        optimizer = self.optimizers()

        self.model.zero_grad()

        train_emb = self.model(train_inputs)
        test_emb = self.model(test_inputs)

        prototypes = self.get_prototypes(train_emb, train_targets, self.num_classes_per_task)
        loss = self.protoypical_loss(prototypes, train_emb, train_targets)

        with torch.no_grad():
            acc = get_proto_accuracy(prototypes, test_emb, test_targets)

#         INFO: use this code for more control over the backward pass
#         loss.backward()
#         optimizer.zero_grad()
#         self.manual_backward(loss, optimizer)
#         optimizer.step()

        self.log_dict({
            "loss": loss.item(),
            "accuracy": acc.item()
        }, prog_bar=True)
        return loss

    def _euclidean_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def cactus_training_step(self, sample, sample_idx):

        # Training step CACTUS-ProtoNets, differs from above default supervised loop
        # TODO: check if both above and below can be merged into one common step

        xs = sample['train'] # support
        xq = sample['test'] # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds.requires_grad_(False)

        if xq.is_cuda:
            target_inds = target_inds.to(self.device)

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.model(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = self._euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        res = {
            'loss': loss_val,
            'acc': acc_val.item()
        }

        if self.training:
            self.log_dict({
                'loss': loss_val.item(),
                'train_acc': acc_val.item()
            }, prog_bar=True)

        return res

    def validation_step(self, batch, batch_idx):

        if self.cactus_flag:
            self.trainer.datamodule.val_dataloader().dataset.reset()
            loss, acc = itemgetter('loss', 'acc')(self.cactus_training_step(batch, batch_idx))

            self.log_dict({
                'val_loss': loss.item(), 'val_acc': acc
            }, prog_bar=True)
        else:
            return -1

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if not self.cactus_flag:
            # this branch should only execute if using normal supervised ProtoNets
            return batch, batch_idx


        # everything below this point is for fixing the batch dimension
        # for the cactus version of protonets

        xs = batch['train'] # support
        xq = batch['test'] # query

        xs.squeeze_(0)
        xq.squeeze_(0)

        batch['train'] = xs
        batch['test'] = xq

        return batch, batch_idx


    def on_train_epoch_end(self, unused=None):
        # doesn't matter if it is a new dataloader object
        # it still points to the same dataset and will correctly hit reset on it
        self.trainer.datamodule.train_dataloader().dataset.reset()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if not self.cactus_flag:
            return batch, batch_idx

        xs = batch['train'] # support
        xq = batch['test'] # query

        xs.squeeze_(0)
        xq.squeeze_(0)

        batch['train'] = xs
        batch['test'] = xq

        return batch, batch_idx
