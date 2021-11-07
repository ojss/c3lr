__all__ = ["protoclr_ae"]

import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import (AdvancedProfiler, PyTorchProfiler,
                                        SimpleProfiler)
from unsupervised_meta_learning.callbacks.confinterval import *
from unsupervised_meta_learning.callbacks.image_generation import *
from unsupervised_meta_learning.callbacks.pcacallbacks import *
from unsupervised_meta_learning.callbacks.umapcallbacks import *
from unsupervised_meta_learning.pl_dataloaders import (UnlabelledDataModule,
                                                       UnlabelledDataset)
from unsupervised_meta_learning.proto_utils import (Decoder, Decoder4L,
                                                    Decoder4L4Mini, Encoder)
from unsupervised_meta_learning.protoclr import ProtoCLR


def protoclr_ae(
    dataset,
    datapath,
    lr=1e-3,
    inner_lr=1e-3,
    gamma=1.0,
    distance="euclidean",
    ckpt_dir=Path("./ckpts"),
    ae=True,
    tau=0.5,
    eval_ways=5,
    clustering_alg="kmeans",
    cluster_on_latent=False,
    eval_support_shots=1,
    eval_query_shots=15,
    n_images=None,
    n_classes=4,
    logging="wandb",
    log_images=False,
    profiler="torch",
    oracle_mode=False,
):

    dm = UnlabelledDataModule(
        dataset,
        datapath,
        split="train",
        transform=None,
        n_support=1,
        n_query=3,
        n_images=n_images,
        n_classes=n_classes,
        batch_size=50,
        seed=10,
        mode="trainval",
        num_workers=0,
        eval_ways=eval_ways,
        eval_support_shots=eval_support_shots,
        eval_query_shots=eval_query_shots,
        train_oracle_mode=oracle_mode,
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
        oracle_mode=oracle_mode,
    )
    dataset_train = UnlabelledDataset(
        dataset=dataset, datapath=datapath, split="train", n_support=1, n_query=3
    )

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
                "oracle_mode": oracle_mode,
                "timestamp": str(datetime.now()),
            },
        )
        logger.watch(model)

        cbs = [
            EarlyStopping(monitor="val_loss", patience=300, min_delta=0.02),
            UMAPCallback(),
            UMAPCallbackOnTrain(every_n_steps=50),
            PCACallback(),
            PCACallbackOnTrain(every_n_steps=50),
            ConfidenceIntervalCallback(),
        ]
        if ae == True:
            cbs.insert(0, WandbImageCallback(get_train_images(dataset_train, 8)))

    elif logging == "tb":
        logger = TensorBoardLogger(save_dir="tb_logs")
        cbs = [TensorBoardImageCallback(get_train_images(dataset_train, 8))]

    ckpt_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        dirpath=ckpt_dir / f"{dataset}/{eval_ways}_{eval_support_shots}_om-{oracle_mode}/{str(datetime.now())}",
        filename="{epoch}-{step}-{val_loss:.2f}-{val_accuracy:.3f}",
        every_n_epochs=30,
        save_top_k=5,
    )
    cbs.append(ckpt_callback)

    if "torch" == profiler:
        profiler = PyTorchProfiler(profile_memory=True, with_stack=True)
    elif "simple" == profiler:
        profiler = SimpleProfiler()
    elif "advanced" == profiler:
        profiler = AdvancedProfiler(dirpath="profiler/")
    if torch.cuda.is_available():
        gpus = -1
    else:
        gpus = None
    trainer = pl.Trainer(
        profiler=profiler,
        max_epochs=1000,
        min_epochs=500,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=15,
        limit_test_batches=600,
        callbacks=cbs,
        num_sanity_val_steps=2,
        gpus=gpus,
        logger=logger,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.test(datamodule=dm, ckpt_dir=ckpt_callback.best_model_path)
    wandb.finish()
