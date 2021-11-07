import importlib
from typing import Any

import plotly.express as px
import pytorch_lightning as pl
import torch
import torch.functional as F
import wandb
from sklearn import cluster

if (cuml_details := importlib.util.find_spec("cuml")) is not None:
    from cuml.manifold import umap
    extra_args = {"verbose": False}
else:
    import umap

class UMAPCallbackOnTrain(pl.Callback):
    def __init__(self, every_n_steps=10) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps

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
            x_query = data[:, :, pl_module.n_support :]
            # e.g. [1,50*n_query,*(3,84,84)]
            x_query = x_query.reshape(
                (batch_size, ways * pl_module.n_query, *x_query.shape[-3:])
            )
            if pl_module.oracle_mode:
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
                n_components=3,
                min_dist=0.5,
                n_neighbors=50,
            )
            z_prime = mapper.fit_transform(z, y=y)

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
                {"UMAP of train embeddings": fig,}, step=trainer.global_step,
            )


class UMAPCallback(pl.Callback):
    # currently only works with wandb
    def __init__(self, every_n_epochs=10, logger="wandb", semi_supervised_umap=False) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.logging_tech = logger
        self.semi_supervised_umap = semi_supervised_umap

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
        labels = torch.cat([y_train.flatten(), y_test.flatten()]).cpu()
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
                n_components=3,
                min_dist=0.1,
                n_neighbors=50,
            ).fit_transform(z)

        if self.logging_tech == "wandb":
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
                {"UMAP of val embeddings": fig,}, step=trainer.global_step,
            )
        elif self.logging_tech == "tb":
            pass


class UMAPClusteringCallback(pl.Callback):
    def __init__(
        self,
        image_f,
        cluster_on_latent=True,
        every_n_epochs=1,
        n_clusters=5,
        cluster_alg="kmeans",
        kernel="rbf",
        logger="wandb",
    ) -> None:
        super().__init__()
        self.image_f = image_f
        self.every_n_epochs = every_n_epochs
        self.logging_tech = logger
        self.cluster_alg = cluster_alg
        self.n_clusters = n_clusters
        self.cluster_on_latent = cluster_on_latent

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            imgs, labels = self.image_f()
            imgs, labels = imgs.to(pl_module.device), labels.to(labels.device)
            with torch.no_grad():
                pl_module.eval()
                z, _ = pl_module(imgs)
                pl_module.train()
            z = F.normalize(z.detach()).cpu().tolist()
            xs = umap.UMAP(random_state=42, n_components=3).fit_transform(z)
            data = z if self.cluster_on_latent == True else xs

            if self.cluster_alg == "kmeans":
                predicted_labels = cluster.KMeans(n_clusters=5).fit_predict(data)
            elif self.cluster_alg == "spectral":
                predicted_labels = cluster.SpectralClustering(n_clusters=5).fit_predict(
                    data
                )

            fig0 = px.scatter_3d(
                x=xs[:, 0],
                y=xs[:, 1],
                z=xs[:, 2],
                color=labels,
                template="seaborn",
                color_discrete_sequence=px.colors.qualitative.Prism,
                color_continuous_scale=px.colors.diverging.Portland,
            )
            fig1 = px.scatter_3d(
                x=xs[:, 0],
                y=xs[:, 1],
                z=xs[:, 2],
                color=predicted_labels,
                template="seaborn",
                color_discrete_sequence=px.colors.qualitative.Prism,
                color_continuous_scale=px.colors.diverging.Portland,
            )
            if self.logging_tech == "wandb":
                wandb.log(
                    {"UMAP clustering of embeddings": fig0,}, step=trainer.global_step,
                )
                wandb.log({"KMeans results": fig1}, step=trainer.global_step)
            elif self.logging_tech == "tb":
                pass
            del xs
            del z
            del data
            del predicted_labels
