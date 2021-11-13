from typing import List 
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import dataset

from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.proto_utils import Decoder4L, AttnEncoder4L, Encoder4L
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule



@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    print("hello")
    train_protoclr_ae(config)


def train_protoclr_ae(
    config
):
    print(config)
    try:
        os.environ["SLURM_JOB_NAME"] = "bash"
        del os.environ["SLURM_NTASKS"]
        del os.environ["SLURM_JOB_NAME"]
    except Exception as e:
        pass
    ckpt_dir = Path("./ckpts")

    print(f"Instantiating datamodule <{config.datamodule._target_}>")
    dm: UnlabelledDataModule = hydra.utils.instantiate(config.datamodule)
    # Init lightning model
    print(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    
    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    print(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, 
        # callbacks=callbacks,
        logger=logger, 
        _convert_="partial"
    )

    log_hyperparameters(
        config=config,
        model=model,
        datamodule=dm,
        trainer=trainer,
        callbacks=[],
        logger=logger
    )
    print("Starting training!")
    trainer.fit(model, datamodule=dm)

     # Get metric score for hyperparameter optimization
    score = trainer.callback_metrics.get(config.get("optimized_metric"))
    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        print("Starting testing!")
        trainer.test(model=model, datamodule=dm, ckpt_path="best")

    print("Finalizing!")
    finish(
        config=config,
        model=model,
        datamodule=dm,
        trainer=trainer,
        callbacks=[],
        logger=logger,
    )
    return score


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()