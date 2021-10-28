__all__ = ['cactus']

import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from unsupervised_meta_learning.cactus import *
from unsupervised_meta_learning.protonets import (CactusPrototypicalModel,
                                                  ProtoModule)


def cactus(
    emb_data_dir: Path = None,
    n_ways=20,
    n_shots=1,
    query=15,
    batch_size=1,
    epochs=300,
    use_precomputed_partitions=False,
    final_chkpt_name="final.chkpt",
    final_chkpt_loc=os.getcwd(),
):

    dm = CactusDataModule(
        ways=n_ways,
        shots=n_shots,
        query=query,
        use_precomputed_partitions=use_precomputed_partitions,
        emb_data_dir=emb_data_dir,
    )
    model = ProtoModule(
        encoder=CactusPrototypicalModel(in_channels=1, hidden_size=64),
        num_classes=20,
        lr=1e-3,
        cactus_flag=True,
    )

    logger = WandbLogger(
        project="protonet",
        config={
            "batch_size": batch_size,
            "steps": 30000,
            "dataset": "omniglot",
            "cactus": True,
            "pre-loaded-partitions": use_precomputed_partitions,
            "partitions": 1,
            "n_ways": n_ways,
            "n_shots": n_shots,
            "query_shots": query,
        },
        log_model=True,
    )

    trainer = pl.Trainer(
        profiler="simple",
        #         max_steps=30_000,
        max_epochs=epochs,
        fast_dev_run=False,
        gpus=1,
        log_every_n_steps=25,
        check_val_every_n_epoch=1,
        flush_logs_every_n_steps=1,
        num_sanity_val_steps=2,
        logger=logger,
        default_root_dir="/home/nfs/oshirekar/unsupervised_ml/cactus_chkpnts/",
        checkpoint_callback=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(os.path.join(final_chkpt_loc, final_chkpt_name))

    wandb.finish()

    return 0
