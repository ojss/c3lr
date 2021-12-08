import pytorch_lightning as pl
import torch
import wandb
from torchvision.utils import make_grid

from unsupervised_meta_learning.pl_dataloaders import OracleDataset, UnlabelledDataset


def get_train_images(ds, num):
    if isinstance(ds, UnlabelledDataset):
        res =  torch.stack([ds[i]["data"][0] for i in range(num)], dim=0)
    elif isinstance(ds, OracleDataset):
        data = ds[0]['data']
        labels = ds[0]['labels']
        
# Divide into support and query shots
        x_support = data[:, :1].reshape(-1, *data.shape[-3: ])
        x_query = data[:, 1:].reshape(-1, *data.shape[-3: ])
        # # e.g. [1,50*n_query,*(3,84,84)]
        x = torch.cat([x_support, x_query])
        res = {
            'x': x,
            'y': labels
        }
    return res


class WandbImageCallback(pl.Callback):
    """
    Logs the input and output images of a module.
    """

    def __init__(self, input_imgs, every_n_epochs=5):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                _, reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=2,)  #  normalize=True, range=(-1,1))
            trainer.logger.experiment.log(
                {
                    "reconstructions": wandb.Image(grid, caption="Reconstructions"),
                    "global_step": trainer.global_step,
                }
            )
            # trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


# Cell
class TensorBoardImageCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=5):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                _, reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=2,)  #  normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step
            )

