# AUTOGENERATED! DO NOT EDIT! File to edit: 03b_ProtoCLR.ipynb (unless otherwise specified).

__all__ = [
    "Classifier",
    "ProtoCLR",
]

# Cell
# export
import copy
import importlib

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from tqdm.auto import tqdm
import pacmap
from unsupervised_meta_learning.dataclasses.protoclr_container import PCLRParamsContainer

from .proto_utils import (AttnEncoder4L, Decoder4L, Encoder4L, cluster_diff_loss,
                          clusterer, get_prototypes, prototypical_loss)

if (cuml_details := importlib.util.find_spec("cuml")) is not None:
    from cuml.manifold import umap

    print("Using CUML for UMAP")
else:
    import umap


# Cell
class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _set_params(self, weight, bias):
        state_dict = dict(weight=weight, bias=bias)
        self.fc.load_state_dict(state_dict)

    def init_params_from_prototypes(self, z_support, n_way, n_support):
        z_support = z_support.contiguous()
        z_proto = z_support.view(n_way, n_support, -1).mean(
            1
        )  # the shape of z is [n_data, n_dim]
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2 * z_proto, bias=-torch.norm(z_proto, dim=-1) ** 2)


# Cell
class ProtoCLR(pl.LightningModule):
    def __init__(
            self,
            params: PCLRParamsContainer
    ):
        super().__init__()
        self.params = params

        self.encoder = params.encoder_class(params.num_input_channels, params.base_channel_size, params.latent_dim)

        self.clustering_algo = params.clustering_algo
        self.cl_reduction = params.cl_reduction
        print(f"Clustering algo in use: {params.clustering_algo}")
        self.ae = params.ae
        if self.ae == True:
            self.decoder = params.decoder_class(
                params.num_input_channels, params.base_channel_size, params.latent_dim
            )
        else:
            self.decoder = nn.Identity()

        self.batch_size = params.batch_size
        self.n_support = params.n_support
        self.n_query = params.n_query

        self.distance = params.distance
        self.tau = params.tau

        # gamma will be used to weight the values of the MSE loss to potentially bring it up to par
        # gamma can also be adaptive in the future
        self.gamma = params.gamma
        self.lr = params.lr
        self.lr_decay_rate = params.lr_decay_rate
        self.lr_decay_step = params.lr_decay_step
        self.inner_lr = params.inner_lr

        self.mode = params.mode
        self.eval_ways = params.eval_ways
        self.sup_finetune = params.sup_finetune
        self.sup_finetune_lr = params.sup_finetune_lr
        self.sup_finetune_epochs = params.sup_finetune_epochs
        self.ft_freeze_backbone = params.ft_freeze_backbone
        self.finetune_batch_norm = params.finetune_batch_norm

        self.log_images = params.log_images
        self.train_oracle_mode = params.train_oracle_mode
        self.train_oracle_ways = params.train_oracle_ways
        self.train_oracle_shots = params.train_oracle_shots

        self.umap = params.use_umap
        self.pacmap = params.use_pacmap
        self.km_clusters = params.km_clusters
        if self.train_oracle_mode is True and params.train_oracle_ways is not None and params.train_oracle_shots is not None:
            self.no_unsqueeze_flg = True
        else:
            self.no_unsqueeze_flg = False

        # self.example_input_array = [batch_size, 1, 28, 28] if dataset == 'omniglot'\
        #     else [batch_size, 3, 84, 84]

        # self.automatic_optimization = False

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.lr_decay_step, gamma=self.lr_decay_rate
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, 'interval': 'step'}}

    def forward(self, x):
        z = self.encoder(x.view(-1, *x.shape[-3:]))
        embeddings = nn.Flatten()(z)
        if isinstance(self.encoder, AttnEncoder4L):
            z = z.unsqueeze(-1).unsqueeze(-1)
            recons = self.decoder(z)
        else:
            recons = self.decoder(z)
        return (
            embeddings.view(*x.shape[:-3], -1),
            recons.view(*x.shape) if self.ae == True else torch.tensor(-1.0),
        )

    def _get_pixelwise_reconstruction_loss(self, x, x_hat, ways):
        mse_loss = (
            F.mse_loss(x.squeeze(0), x_hat.squeeze(0), reduction="none").sum(
                dim=[
                    1,
                    2,
                    3,
                ]
            ).mean(dim=[0])
        )
        return mse_loss

    def calculate_protoclr_loss(self, z, y_support, y_query, ways):

        #
        # e.g. [1,50*n_support,*(3,84,84)]
        z_support = z[:, : ways * self.n_support]
        # e.g. [1,50*n_query,*(3,84,84)]
        z_query = z[:, ways * self.n_support:]
        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support  # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        loss, accuracy = prototypical_loss(
            z_proto, z_query, y_query, distance=self.distance, tau=self.tau
        )
        return loss, accuracy

    def _get_cluster_loss(self, z: torch.Tensor, y_support, y_query, ways):
        tau = self.tau
        loss = 0.0
        emb_list = F.normalize(z.squeeze(0).detach()).cpu().numpy()
        if self.train_oracle_mode:
            y = torch.cat([y_support, y_query], dim=0).detach().cpu().numpy()
        else:
            y = torch.cat([y_support, y_query], dim=1).detach().cpu().flatten().numpy()

        #
        # e.g. [50*n_support,2]
        z_support = z[
                    :, : ways * self.n_support, :
                    ]  # TODO: make use of this in the loss somewhere?
        # e.g. [50*n_query,2]
        z_query = z[:, ways * self.n_support:, :]
        if self.train_oracle_mode:
            loss = cluster_diff_loss(
                z,
                y,
                self.eval_ways,
                similarity=self.distance,
                temperature=tau,
                reduction=self.cl_reduction
            )
        else:
            if self.umap == True:
                reduced_z = umap.UMAP(
                    random_state=self.params.seed,
                    n_components=3,
                    min_dist=0.25,
                    n_neighbors=50
                ).fit_transform(
                    emb_list, 
                    y=y
                )  # (n_samples, 3)
            elif self.pacmap == True:
                reduced_z = pacmap.PaCMAP(n_dims=3, n_neighbours=50).fit_transform(emb_list)
            else:
                reduced_z = emb_list # technically not reduced
            if self.clustering_algo == "kmeans":
                clf, predicted_labels, _ = clusterer(reduced_z, n_clusters=self.km_clusters, algo="kmeans")
                loss = cluster_diff_loss(
                    z,
                    predicted_labels,
                    similarity=self.distance,
                    temperature=tau,
                    reduction=self.cl_reduction
                )
            elif self.clustering_algo == "hdbscan":
                clf, predicted_labels, probs = clusterer(
                    reduced_z, algo="hdbscan", hdbscan_metric="euclidean"
                )
                predicted_labels = torch.from_numpy(predicted_labels).type_as(z)
                if -1 in predicted_labels:
                    # breakpoint()
                    non_noise_indices = ~(predicted_labels == -1)
                    self.log('noise_count', torch.where(predicted_labels == -1)[0].shape[0])
                    predicted_labels = predicted_labels.masked_select(
                        non_noise_indices
                    )
                    z = z.index_select(
                        1, non_noise_indices.nonzero().flatten()
                    )

                loss = cluster_diff_loss(
                    z,
                    predicted_labels,
                    similarity=self.distance,
                    temperature=tau,
                    reduction=self.cl_reduction
                )

        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        # opt.zero_grad()
        # [batch_size x ways x shots x image_dim]
        data = batch["data"]
        if not self.no_unsqueeze_flg:
            data = data.unsqueeze(0)
        # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
        batch_size = data.size(0)
        ways = data.size(1)

        # Divide into support and query shots
        x_support = data[:, :, : self.n_support]
        # e.g. [1,50*n_support,*(3,84,84)]
        x_support = x_support.reshape(
            (batch_size, ways * self.n_support, *x_support.shape[-3:])
        )
        x_query = data[:, :, self.n_support:]
        # e.g. [1,50*n_query,*(3,84,84)]
        x_query = x_query.reshape(
            (batch_size, ways * self.n_query, *x_query.shape[-3:])
        )

        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(0).unsqueeze(2)  # batch and shot dim
        y_query = y_query.repeat(batch_size, 1, self.n_query)
        y_query = y_query.view(batch_size, -1).type_as(data).long()

        y_support = torch.arange(ways).unsqueeze(0).unsqueeze(2)  # batch and shot dim
        y_support = y_support.repeat(batch_size, 1, self.n_support)
        y_support = y_support.view(batch_size, -1).type_as(data).long()

        # Extract features (first dim is batch dim)>
        # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        x = torch.cat([x_support, x_query], 1)
        z, x_hat = self.forward(x)

        loss, accuracy = self.calculate_protoclr_loss(z, y_support, y_query, ways)
        self.log_dict({"clr_loss": loss.item()}, prog_bar=True)

        if self.train_oracle_mode and self.clustering_algo is None:
            # basically leaking info to check if things work in our favor
            # works only for omniglot at the moment
            labels = batch["labels"]
            if not self.no_unsqueeze_flg:
                # TODO: see if these .cpu calls need to go
                y_support = labels[:, 0].cpu()
                y_query = labels[:, 1:].flatten().cpu()
                lb_enc = LabelEncoder()
                lb_enc.fit(y_support)
                y_support = torch.Tensor(lb_enc.transform(y_support)).type_as(labels)
                y_query = torch.Tensor(lb_enc.transform(y_query)).type_as(labels)
            else:
                labels = labels.squeeze(0)
                y_support = labels[:self.train_oracle_shots * self.train_oracle_ways]
                y_query = y_support.repeat_interleave(3)
            loss_cluster = self._get_cluster_loss(z, y_support, y_query, ways)
            self.log("cluster_clr", loss_cluster.item(), prog_bar=True)
            loss += loss_cluster

        elif self.clustering_algo is not None:
            loss_cluster = self._get_cluster_loss(z, y_support, y_query, ways)
            self.log("cluster_clr", loss_cluster.item(), prog_bar=True)
            loss += loss_cluster

        # adding the pixelwise reconstruction loss at the end
        # it has been broadcasted such that each support source image is broadcasted thrice over the three
        # query set images - which are the augmentations of the support image
        if self.ae:
            mse_loss = (
                    self._get_pixelwise_reconstruction_loss(x, x_hat, ways) * self.gamma
            )
            self.log(
                "mse_loss",
                mse_loss.item(),
                prog_bar=True
            )
            loss += mse_loss
        torch.cuda.synchronize()
        torch.autograd.set_detect_anomaly(True)

        # self.manual_backward(loss)
        # opt.step()
        # sch.step()

        self.log_dict({"loss": loss.item(), "train_accuracy": accuracy}, prog_bar=True)

        return {"loss": loss, "train_accuracy": accuracy}

    @torch.enable_grad()
    def supervised_finetuning(
            self,
            encoder,
            episode,
            proto_init=True,
            freeze_backbone=False,
            finetune_batch_norm=False,
            inner_lr=0.001,
            total_epoch=15,
            n_way=5,
    ):
        x_support = episode["train"][0][0]  # only take data & only first batch
        x_support = x_support
        x_support_var = Variable(x_support)
        x_query = episode["test"][0][0]  # only take data & only first batch
        x_query = x_query
        x_query_var = Variable(x_query)
        n_support = x_support.shape[0] // n_way
        n_query = x_query.shape[0] // n_way

        batch_size = n_way
        support_size = n_way * n_support

        y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).type_as(
            x_support
        )  # (25,)

        x_b_i = x_query_var
        x_a_i = x_support_var
        encoder.eval()

        z_a_i = nn.Flatten()(encoder(x_a_i))  # .view(*x_a_i.shape[:-3], -1)
        encoder.train()

        # Define linear classifier
        input_dim = z_a_i.shape[1]
        classifier = Classifier(input_dim, n_way=n_way).to(self.device)
        classifier.train()
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss()
        # Initialise as distance classifer (distance to prototypes)
        if proto_init:
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support)
        classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)
        if freeze_backbone is False:
            delta_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr
            )
        # Finetuning
        if freeze_backbone is False:
            encoder.train()
        else:
            encoder.eval()
        classifier.train()
        if not finetune_batch_norm:
            for module in encoder.modules():
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()

        for epoch in tqdm(range(total_epoch), total=total_epoch, leave=False):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy(
                    rand_id[j: min(j + batch_size, support_size)]
                ).type_as(x_support).long()

                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id].long()
                #####################################

                output = nn.Flatten()(encoder(z_batch))
                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
        classifier.eval()
        encoder.eval()

        output = nn.Flatten()(encoder(x_b_i))
        scores = classifier(output)

        y_query = torch.tensor(np.repeat(range(n_way), n_query)).type_as(x_support).long()
        loss = F.cross_entropy(scores, y_query, reduction="mean")
        _, predictions = torch.max(scores, dim=1)
        accuracy = torch.mean(predictions.eq(y_query).float())
        return loss, accuracy.item()

    def validation_step(self, batch, batch_idx):

        original_encoder_state = copy.deepcopy(self.encoder.state_dict())
        if not self.mode == "trainval":
            original_encoder_state = copy.deepcopy(self.encoder.state_dict())

        if self.sup_finetune:
            loss, accuracy = self.supervised_finetuning(
                self.encoder,
                episode=batch,
                inner_lr=self.sup_finetune_lr,
                total_epoch=self.sup_finetune_epochs,
                freeze_backbone=self.ft_freeze_backbone,
                finetune_batch_norm=self.finetune_batch_norm,
                n_way=self.eval_ways,
            )
            self.encoder.load_state_dict(original_encoder_state)
        elif self.mode == "trainval":
            with torch.no_grad():
                loss, accuracy, _, _ = self.calculate_protoclr_loss(batch, ae=False)
        else:
            with torch.no_grad():
                loss, accuracy, _, _ = self.calculate_protoclr_loss(batch, ae=False)
        self.log_dict(
            {"val_loss": loss.item(), "val_accuracy": accuracy}, prog_bar=True, sync_dist=True
        )
        return loss.item(), accuracy

    def test_step(self, batch, batch_idx):
        original_encoder_state = copy.deepcopy(self.encoder.state_dict())
        # if self.sup_finetune:
        loss, accuracy = self.supervised_finetuning(
            self.encoder,
            episode=batch,
            inner_lr=self.sup_finetune_lr,
            total_epoch=self.sup_finetune_epochs,
            freeze_backbone=self.ft_freeze_backbone,
            finetune_batch_norm=self.finetune_batch_norm,
            n_way=self.eval_ways,
        )
        self.encoder.load_state_dict(original_encoder_state)
        self.log(
            "test_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            "test_acc",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss.item(), accuracy
