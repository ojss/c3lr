import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import wandb
from scipy import stats


class ConfidenceIntervalCallback(pl.Callback):
    def __init__(self, plot=True) -> None:
        super().__init__()
        self.losses = []
        self.accuracies = []
        self.plot = plot

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        loss, accuracy = outputs
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def on_test_end(self, trainer, pl_module) -> None:
        conf_interval = stats.t.interval(
            0.95,
            len(self.accuracies) - 1,
            loc=np.mean(self.accuracies),
            scale=stats.sem(self.accuracies),
        )
        if self.plot:
            wandb.log({"Confidence Interval": conf_interval}, commit=False)

            plt.ylabel("Average Test Accuracy")
            plt.errorbar(
                [1],
                np.mean(self.accuracies),
                yerr=np.std(self.accuracies),
                fmt="o",
                color="black",
                ecolor="lightgray",
                elinewidth=3,
                capsize=0,
            )
            wandb.log(
                {"Average Test Accuracy with std dev": wandb.Image(plt)}, commit=False
            )
        else:
            print(f"Confidence Interval: {conf_interval}")
            print(
                f"Accuracy: {100 * np.mean(self.accuracies):.2f} +- {100 * (conf_interval[1] - np.mean(self.accuracies)):.2f}"
            )
        return conf_interval
