import os
import warnings
from datetime import datetime
from pathlib import Path
from functools import partial

import fire
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
from unsupervised_meta_learning.cactus import *
from unsupervised_meta_learning.pl_dataloaders import (
    UnlabelledDataModule,
    UnlabelledDataset,
    get_episode_loader,
)
from unsupervised_meta_learning.proto_utils import (
    Decoder4L,
    Decoder4L4Mini,
    Encoder,
    Decoder,
    get_images_labels_from_dl,
)
from unsupervised_meta_learning.protoclr import (
    ConfidenceIntervalCallback,
    ProtoCLR,
    TensorBoardImageCallback,
    UMAPCallback,
    UMAPClusteringCallback,
    WandbImageCallback,
    get_train_images,
)
from unsupervised_meta_learning.protonets import CactusPrototypicalModel, ProtoModule


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


def protoclr_ae(
    dataset,
    datapath,
    lr=1e-3,
    inner_lr=1e-3,
    gamma=1.0,
    distance="euclidean",
    ckpt_dir='./ckpts',
    ae=True,
    tau=0.5,
    eval_ways=5,
    clustering_alg="kmeans",
    cluster_on_latent=False,
    eval_support_shots=1,
    eval_query_shots=15,
    logging="wandb",
    log_images=False,
):

    dm = UnlabelledDataModule(
        dataset,
        datapath,
        split="train",
        transform=None,
        n_support=1,
        n_query=3,
        n_images=None,
        n_classes=None,
        batch_size=50,
        seed=10,
        mode="trainval",
        eval_ways=eval_ways,
        eval_support_shots=eval_support_shots,
        eval_query_shots=eval_query_shots,
    )

    if dataset == "omniglot":
        decoder_class = Decoder4L
        num_input_channels = 1
    elif dataset == "miniimagenet":
        decoder_class = Decoder4L4Mini
        num_input_channels = 3
    model = ProtoCLR(
        n_support=1,
        n_query=3,
        batch_size=50,
        gamma=gamma,
        lr=lr,
        inner_lr=inner_lr,
        lr_decay_step=25000,
        lr_decay_rate=0.5,
        decoder_class=decoder_class,
        num_input_channels=num_input_channels,
        distance=distance,
        τ=tau,
        ae=ae,
        clustering_algo=clustering_alg,
        log_images=log_images,
    )
    dataset_train = UnlabelledDataset(
        dataset=dataset, datapath=datapath, split="train", n_support=1, n_query=3
    )
    dl = get_episode_loader(
        dataset,
        datapath,
        ways=5,
        shots=5,
        test_shots=15,
        batch_size=1,
        split="val",
    )
    image_f = partial(get_images_labels_from_dl, dl)

    if logging == "wandb":
        logger = WandbLogger(
            project="ProtoCLR+AE",
            config={
                "batch_size": 50,
                "steps": 100,
                "lr": lr,
                "inner_lr": inner_lr,
                "gamma": gamma,
                "ae": ae,
                "τ": tau,
                "distance": distance,
                "dataset": dataset,
                "eval_ways": eval_ways,
                "eval_support_shots": eval_support_shots,
                "eval_query_shots": eval_query_shots,
                "clustering_algo": clustering_alg,
                "clustering_on_latent": cluster_on_latent,
                "timestamp": str(datetime.now()),
            },
        )
        logger.watch(model)

        cbs = [
            EarlyStopping(monitor="val_loss", patience=300, min_delta=0.02),
            UMAPClusteringCallback(
                image_f,
                every_n_epochs=1,
                cluster_alg=clustering_alg,
                cluster_on_latent=cluster_on_latent,
            ),
            ConfidenceIntervalCallback(),
        ]
        if ae == True:
            cbs.insert(0, WandbImageCallback(get_train_images(dataset_train, 8)))

    elif logging == "tb":
        logger = TensorBoardLogger(save_dir="tb_logs")
        cbs = [TensorBoardImageCallback(get_train_images(dataset_train, 8))]

    cbs.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
            every_n_epochs=100,
        )
    )

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=10000,
        min_epochs=500,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=15,
        limit_test_batches=600,
        callbacks=cbs,
        num_sanity_val_steps=2,
        gpus=-1,
        logger=logger,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.test()
    wandb.finish()


if __name__ == "__main__":
    fire.Fire({"cactus": cactus, "protoclr_ae": protoclr_ae})
