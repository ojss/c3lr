# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import nn

import hdbscan
from . import re_rank


def fast_grouped_mean(z, labels):
    tmp = labels.view(labels.size(0), 1).expand(-1, z.squeeze(0).size(1))
    unique_labels, labels_count = tmp.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, tmp, z.squeeze(0))
    res = res / labels_count.float().unsqueeze(1)

    return res

class ProtoCLR(nn.Module):
    """Calculate the UMTRA-style loss on a batch of images.
    If shots=1 and only two views are served for each image,
    this corresponds exactly to UMTRA except that it uses ProtoNets
    instead of MAML.

    Parameters:
        - model_func: The encoder network.
        - shots: The number of support shots.
    """
    def __init__(self, model_func, shots=1):
        super(ProtoCLR, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.feature    = model_func()
        self.top1 = utils.AverageMeter()
        self.shots = shots

    def forward(self, x):
        clr_loss = torch.tensor(0)
        # Treat the first dim as way, the second as shots
        ways = x.size(0)
        n_views = x.size(1)
        shots = self.shots
        query_shots = n_views - shots
        x_support = x[:,:shots].reshape((ways*shots, *x.shape[-3:]))
        x_support = Variable(x_support.cuda())
        x_query = x[:,shots:].reshape((ways*query_shots, *x.shape[-3:]))
        x_query = Variable(x_query.cuda())

        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(1) # shot dim
        y_query = y_query.repeat(1, query_shots)
        y_query = y_query.view(-1).cuda()

        # Extract features
        x_both = torch.cat([x_support, x_query], 0)
        z = self.feature(x_both)
        z_support = z[:ways*shots]
        z_query = z[ways*shots:]

        # Get prototypes
        z_proto = z_support.view(ways, shots, -1).mean(1) #the shape of z is [n_data, n_dim]
        z_numpy = z.squeeze(0).detach().cpu().numpy()
        kjrd_d = re_rank.re_ranking2(z_numpy, z_numpy, k1=20, k2=6, lambda_value=0)
        clf = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5, core_dist_n_jobs=4)
        clf.fit(kjrd_d.astype(float))
        predicted_labels = torch.from_numpy(clf.labels_).type_as(z)
        if -1 in predicted_labels:
            non_noise_indices = ~(predicted_labels == -1)
            print(f'noise_count: {torch.where(predicted_labels == -1)[0].shape[0]}')
            predicted_labels = predicted_labels.masked_select(
                non_noise_indices
            )
            z = z.index_select(
                0, non_noise_indices.nonzero().flatten()
            )
        if len(predicted_labels) != 0:
            res = fast_grouped_mean(z, predicted_labels.long())
            clr_dists = euclidean_dist(res, z).unsqueeze(0)
            clr_loss = F.cross_entropy(-clr_dists, predicted_labels.unsqueeze(0).long())
        else:
            clr_loss = torch.tensor(0)

        # Calculate loss and accuracies
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores, y_query, clr_loss

    def forward_loss(self, x):
        scores, y, clr_loss = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y) + clr_loss
   
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
 

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
