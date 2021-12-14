import importlib
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import plotly.express as px
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.functional as F
import wandb

from sklearn.manifold import TSNE

from ..common.utils import log_plotly_graph, log_sns_plot
from ..proto_utils import clusterer

if (cuml_details := importlib.util.find_spec("cuml")) is not None:
    from cuml.manifold import umap

    extra_args = {"verbose": False}
else:
    import umap

__all__ = ['UMAPCallbackOnTrain', 'UMAPCallback', 'UMAPConstantInput', 'UMAPClusteringCallback']


class UMAPCallbackOnTrain(pl.Callback):
    def __init__(self, logging_tech="wandb", every_n_steps=10, use_plotly=True) -> None:
        super().__init__()
        self.logging_tech = logging_tech
        self.every_n_steps = every_n_steps
        self.plotly = use_plotly

    def on_train_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if trainer.global_step % self.every_n_steps == 0:
            data = batch["data"]
            data = data.unsqueeze(0)
            # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
            batch_size = data.size(0)
            ways = data.size(1)

            # Divide into support and query shots
            x_support = data[:, :, : pl_module.n_support]
            # e.g. [1,50*n_support,*(3,84,84)]
            x_support = x_support.reshape(
                (batch_size, ways * pl_module.n_support, *x_support.shape[-3:])
            )
            x_query = data[:, :, pl_module.n_support:]
            # e.g. [1,50*n_query,*(3,84,84)]
            x_query = x_query.reshape(
                (batch_size, ways * pl_module.n_query, *x_query.shape[-3:])
            )
            if pl_module.train_oracle_mode:
                labels = batch["labels"]
                y_support = labels[:, 0]
                y_query = labels[:, 1:].flatten()
                y = torch.cat([y_support, y_query]).cpu().numpy()
            else:
                y_query = (
                    torch.arange(ways).unsqueeze(0).unsqueeze(2)
                )  # batch and shot dim
                y_query = y_query.repeat(batch_size, 1, pl_module.n_query)
                y_query = y_query.view(batch_size, -1).type_as(x_query).flatten()

                y_support = (
                    torch.arange(ways).unsqueeze(0).unsqueeze(2)
                )  # batch and shot dim
                y_support = y_support.repeat(batch_size, 1, pl_module.n_support)
                y_support = y_support.view(batch_size, -1).type_as(x_support).flatten()
                y = torch.cat([y_support, y_query]).cpu().numpy()

            x = torch.cat([x_support, x_query], dim=1)

            with torch.no_grad():
                pl_module.eval()
                z, _ = pl_module(x)
                pl_module.train()
            z = z.detach().squeeze(0).cpu().numpy()
            mapper = umap.UMAP(
                random_state=42,
                n_components=3 if self.plotly is True else 2,
                min_dist=0.5,
                n_neighbors=50,
            )
            z_prime = mapper.fit_transform(z, y=y)
            if self.logging_tech == "wandb" and self.plotly:
                fig = px.scatter_3d(
                    x=z_prime[:, 0],
                    y=z_prime[:, 1],
                    z=z_prime[:, 2],
                    color=y,
                    template="seaborn",
                    size_max=18,
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    color_continuous_scale=px.colors.diverging.Portland,
                )
                wandb.log(
                    {
                        "UMAP of train embeddings": fig,
                    },
                    step=trainer.global_step,
                )
            elif self.logging_tech == "wandb" and not self.plotly:
                sns.set_theme()
                ax = sns.scatterplot(
                    x=z_prime[:, 0],
                    y=z_prime[:, 1],
                    hue=y,
                    palette="icefire",
                    style=y,
                    legend=False,
                )
                wandb.log(
                    {"UMAP of train embeddings": wandb.Image(ax)},
                    step=trainer.global_step,
                )
                plt.clf()


class UMAPCallback(pl.Callback):
    # currently only works with wandb
    def __init__(
            self,
            every_n_epochs=10,
            logger="wandb",
            semi_supervised_umap=False,
            use_plotly=True,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.logging_tech = logger
        self.semi_supervised_umap = semi_supervised_umap
        self.plotly = use_plotly

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        x_train, y_train = batch["train"]
        x_test, y_test = batch["test"]

        x = torch.cat([x_train, x_test], dim=1)  # [1, shots * ways, img_shape]
        labels = torch.cat([y_train.flatten(), y_test.flatten()]).cpu().numpy()
        with torch.no_grad():
            pl_module.eval()
            z, _ = pl_module(x)

        z = z.detach().cpu().squeeze(0).numpy()

        if self.semi_supervised_umap:
            # use the pseduo labels only for the support set elements
            # TODO: then use this information to derive better prototypes - is this possible?
            # To test the theory: one way to do it is plot the UMAPped points and log them
            # check the separation and run clustering with HDBSCAN and pull out representative points of
            # the clusters

            pass
        else:
            # running in unsupervised mode
            z_prime = umap.UMAP(
                random_state=42,
                n_components=3 if self.plotly is True else 2,
                min_dist=0.1,
                n_neighbors=50,
            ).fit_transform(z)

        if self.logging_tech == "wandb" and self.plotly:
            fig = px.scatter_3d(
                x=z_prime[:, 0],
                y=z_prime[:, 1],
                z=z_prime[:, 2],
                color=labels,
                template="seaborn",
                size_max=18,
                color_discrete_sequence=px.colors.qualitative.Prism,
                color_continuous_scale=px.colors.diverging.Portland,
            )
            wandb.log(
                {
                    "UMAP of val embeddings": fig,
                },
                step=trainer.global_step,
            )
        elif self.logging_tech == "wandb" and not self.plotly:
            sns.set_theme()
            ax = sns.scatterplot(
                x=z_prime[:, 0],
                y=z_prime[:, 1],
                hue=labels,
                palette="icefire",
                style=labels,
                legend=False,
            )
            wandb.log(
                {
                    "UMAP of val embeddings": wandb.Image(ax),
                },
                step=trainer.global_step,
            )
            plt.clf()
        elif self.logging_tech == "tb":
            pass


class UMAPConstantInput(pl.Callback):
    def __init__(
            self,
            logging_tech="wandb",
            every_n_steps=90,
            input_images=None,
            use_plotly=True,
            clustering="hdbscan",
            km_n_clusters=5,
            cluster_on_latent=False,
    ) -> None:
        super().__init__()
        self.logging_tech = logging_tech
        self.every_n_steps = every_n_steps
        self.input_images = torch.cat([input_images['train'][0], input_images['test'][0]], dim=1)
        # self.input_images = input_images['train'][0]
        self.input_labels = torch.cat([input_images['train'][1].squeeze(0), input_images['test'][1].squeeze(0)]).cpu().numpy()
        # self.input_labels = input_images['train'][1].squeeze(0).cpu().numpy()
        self.plotly = use_plotly
        self.algo = clustering
        self.clusterer = partial(clusterer, algo=clustering, n_clusters=km_n_clusters)
        self.cluster_on_latent = cluster_on_latent

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        input_imgs = self.input_images.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            # these set of input images do not change
            z, _ = pl_module(input_imgs)
            pl_module.train()
        z = z.detach().squeeze(0).cpu().numpy()
        z_prime = umap.UMAP(
            random_state=42,
            n_components=2,
            # min_dist=pl_module.params.umap_min_dist,
            # n_neighbors=pl_module.params.rdim_n_neighbors,
        ).fit_transform(z)
        _, preds, _ = self.clusterer(z if self.cluster_on_latent else z_prime)
        if self.plotly:
            log_plotly_graph(z_prime, self.input_labels, "Raw embeddings of constant images", trainer.global_step,
                             pl_module, dims=2)
            log_plotly_graph(z_prime, preds, "HDBSCAN results", trainer.global_step, pl_module, dims=2)
        return


class UMAPClusteringCallback(pl.Callback):
    def __init__(
            self,
            use_umap=True,
            logging_tech="wandb",
            every_n_steps=90,
            use_plotly=True,
            clustering="hdbscan",
            km_n_clusters=5,
            cluster_on_latent=False,
    ) -> None:
        super().__init__()
        self.logging_tech = logging_tech
        self.every_n_steps = every_n_steps
        self.plotly = use_plotly
        self.algo = clustering
        self.clusterer = partial(clusterer, algo=clustering, n_clusters=km_n_clusters)
        self.cluster_on_latent = cluster_on_latent

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:

        if trainer.global_step % self.every_n_steps == 0:
            data = batch["data"]
            if not pl_module.no_unsqueeze_flg:
                data = data.unsqueeze(0)
            # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
            batch_size = data.size(0)
            ways = data.size(1)

            # Divide into support and query shots
            x_support = data[:, :, : pl_module.n_support]
            # e.g. [1,50*n_support,*(3,84,84)]
            x_support = x_support.reshape(
                (batch_size, ways * pl_module.n_support, *x_support.shape[-3:])
            )
            x_query = data[:, :, pl_module.n_support:]
            # e.g. [1,50*n_query,*(3,84,84)]
            x_query = x_query.reshape(
                (batch_size, ways * pl_module.n_query, *x_query.shape[-3:])
            )
            if pl_module.train_oracle_mode:
                labels = batch["labels"]
                y_support = labels[:, 0]
                y_query = labels[:, 1:].flatten()
                true_y = torch.cat([y_support, y_query]).cpu().numpy()

            y_query = torch.arange(ways).unsqueeze(0).unsqueeze(2)  # batch and shot dim
            y_query = y_query.repeat(batch_size, 1, pl_module.n_query)
            y_query = y_query.view(batch_size, -1).type_as(x_query).flatten()

            y_support = (
                torch.arange(ways).unsqueeze(0).unsqueeze(2)
            )  # batch and shot dim
            y_support = y_support.repeat(batch_size, 1, pl_module.n_support)
            y_support = y_support.view(batch_size, -1).type_as(x_support).flatten()
            y = torch.cat([y_support, y_query]).cpu().numpy()

            x = torch.cat([x_support, x_query], dim=1)

            with torch.no_grad():
                pl_module.eval()
                z, _ = pl_module(x)
                pl_module.train()
            z = z.detach().squeeze(0).cpu().numpy()
            mapper = umap.UMAP(
                random_state=42,
                n_components=pl_module.params.rdim_components,
                min_dist=pl_module.params.umap_min_dist,
                n_neighbors=pl_module.params.rdim_n_neighbors,
            )
            z_prime = mapper.fit_transform(z, y=y)

            _, preds, _ = self.clusterer(z if self.cluster_on_latent else z_prime)

            if self.logging_tech == "wandb" and self.plotly:
                log_plotly_graph(
                    z_prime,
                    preds,
                    f"{self.algo} predictions on train embeddings",
                    trainer.global_step,
                )
                if pl_module.train_oracle_mode:
                    log_plotly_graph(
                        z_prime,
                        true_y if pl_module.train_oracle_mode else y,
                        "UMAP of source data",
                        trainer.global_step,
                    )
            elif self.logging_tech == "wandb" and not self.plotly:
                log_sns_plot(
                    z_prime,
                    preds,
                    f"{self.algo} predictions on train embeddings",
                    trainer.global_step,
                )
                if pl_module.train_oracle_mode:
                    log_sns_plot(
                        z_prime,
                        true_y if pl_module.train_oracle_mode else y,
                        "UMAP of source data",
                        trainer.global_step,
                    )
        return outputs
