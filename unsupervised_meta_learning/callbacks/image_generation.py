import pytorch_lightning as pl
import torch
import wandb
from torchvision.utils import make_grid


def get_train_images(ds, num):
    return torch.stack([ds[i]["data"][0] for i in range(num)], dim=0)


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

