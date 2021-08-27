# AUTOGENERATED! DO NOT EDIT! File to edit: 01d_proto_utils.ipynb (unless otherwise specified).

__all__ = ['euclidean_distance', 'cosine_similarity', 'get_num_samples', 'get_prototypes', 'prototypical_loss',
           'CNN_4Layer']

# Cell
#export
# adapted from the torchmeta code
# TODO: use this in vanilla protonet code
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import conv3x3

# Cell
def euclidean_distance(x, y):
    """
    x, y have shapes (batch_size, num_examples, embedding_size).
    x is prototypes, y are embeddings in most cases
    """
    return torch.sum((x.unsqueeze(2) - y.unsqueeze(1))** 2, dim=-1)

# Cell

def cosine_similarity(x, y):
    """x, y have shapes (batch_size, num_examples, embedding_size)."""

    # compute dot prod similarity x_i.T y_i (numerator)
    dot_similarity = torch.bmm(x, y.permute(0, 2, 1))

    # compute l2 norms ||x_i|| * ||y_i||
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    y_norm = y.norm(p=2, dim=-1, keepdim=True)

    norms = torch.bmm(x, y.permute(0, 2, 1)) + 1e-8

    return dot_similarity / norms


# Cell

def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples

# Cell

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

# Cell
def prototypical_loss(prototypes, embeddings, targets,
                      distance='euclidean', **kwargs):
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

    distance : `String`
        The distance measure to be used: 'eucliden' or 'cosine'

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    if distance == 'euclidean':
        squared_distances = euclidean_distance(prototypes, embeddings)
        loss = F.cross_entropy(-squared_distances, targets, **kwargs)
        _, predictions = torch.min(squared_distances, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    elif distance == 'cosine':
        cosine_similarities = cosine_similarity(prototypes, embeddings)
        loss = F.cross_entropy(cosine_similarities, targets, **kwargs)
        _, predictions = torch.max(cosine_similarities, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    else:
        raise ValueError('Distance must be "euclidean" or "cosine"')
    return loss, accuracy.item()

# Cell
class CNN_4Layer(nn.Module):
    def __init__(self, in_channels, out_channels=64, hidden_size=64):
        super(CNN_4Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )

#         self.unpool = nn.MaxUnpool2d(2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[-3:]))
#         x = self.unpool(embeddings, indices)
        x = self.decoder(embeddings)
        return embeddings.view(*inputs.shape[:-3], -1), x