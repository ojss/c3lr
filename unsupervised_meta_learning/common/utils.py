import plotly.express as px
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import wandb

__all__ = ["log_plotly_graph", "log_sns_plot"]

def log_plotly_graph(z, y, title, global_step, pl_module: pl.LightningModule, dims=2):
    if dims == 3:
        fig = px.scatter_3d(
            x=z[:, 0],
            y=z[:, 1],
            z=z[:, 2],
            color=y,
            template="seaborn",
            size_max=18,
            color_discrete_sequence=px.colors.qualitative.Prism,
            color_continuous_scale=px.colors.diverging.Portland,
        )
    elif dims == 2:
        fig = px.scatter(
            x=z[:, 0],
            y=z[:, 1],
            color=y,
            template="seaborn",
            size_max=18,
            color_discrete_sequence=px.colors.qualitative.Prism,
            color_continuous_scale=px.colors.diverging.Portland,
        )
    wandb.log(
        {title: fig},
        step=global_step,
    )
    return 0

def log_sns_plot(z, y, title, global_step):
    sns.set_theme()
    ax = sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=y, palette="icefire", style=y, legend=False)
    wandb.log(
        {title: wandb.Image(ax)},
        step=global_step,
    )
    plt.clf()
    return 0