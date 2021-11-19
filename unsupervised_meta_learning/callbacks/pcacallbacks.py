import logging
from typing import Any

import plotly.express as px
import pytorch_lightning as pl
import torch
import wandb
import seaborn as sns
import matplotlib.pyplot as plt


class PCACallbackOnTrain(pl.Callback):
    def __init__(self, every_n_steps=10, use_plotly=True, logging_tech='wandb') -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.plotly = use_plotly
        self.logging_tech = logging_tech
    
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
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
            if pl_module.train_oracle_mode:
                labels = batch['labels']
                y_support = labels[:, 0]
                y_query = labels[:, 1:].flatten()
                y = torch.cat([y_support, y_query]).cpu().numpy()
            else:
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
            z = z.detach().squeeze(0).cpu()
            U, S, V = torch.pca_lowrank(z)
            z_prime = z @ V[:, :3 if self.plotly == True else 2]
            z_prime = z_prime.cpu().numpy()

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

                wandb.log({"PCA of train embeddings": fig}, step=trainer.global_step)
            elif self.logging_tech == 'wandb' and not self.plotly:
                sns.set_theme()
                ax = sns.scatterplot(x=z_prime[:, 0], y=z_prime[:, 1], hue=y, palette="icefire", style=y, legend=False)
                wandb.log(
                    {"PCA of train embeddings": wandb.Image(ax),}, step=trainer.global_step,
                )
                plt.clf()

class PCACallback(pl.Callback):
    def __init__(self, every_n_steps=10, use_plotly=True, logging_tech='wandb'):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.plotly = use_plotly
        self.logging_tech = logging_tech

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
        z = z.detach().squeeze(0)
        U, S, V = torch.pca_lowrank(z)
        z_prime = z @ V[:, :3 if self.plotly == True else 2]
        z_prime = z_prime.cpu().numpy()

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

            wandb.log({"PCA of val embeddings": fig}, step=trainer.global_step)
        elif self.logging_tech == 'wandb' and not self.plotly:
            sns.set_theme()
            ax = sns.scatterplot(x=z_prime[:, 0], y=z_prime[:, 1], hue=labels, palette="icefire", style=labels, legend=False)
            wandb.log(
                {"PCA of val embeddings": wandb.Image(ax),}, step=trainer.global_step,
            )
            plt.clf()
