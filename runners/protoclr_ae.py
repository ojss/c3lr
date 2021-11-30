__all__ = ["protoclr_ae"]

import os
import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler, SimpleProfiler

from unsupervised_meta_learning.callbacks.confinterval import *
from unsupervised_meta_learning.callbacks.image_generation import *
from unsupervised_meta_learning.callbacks.pcacallbacks import *
from unsupervised_meta_learning.callbacks.umapcallbacks import *
from unsupervised_meta_learning.pl_dataloaders import (
    UnlabelledDataModule,
    UnlabelledDataset,
    OracleDataModule,
)
from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.dataclasses.protoclr_container import (
    PCLRParamsContainer,
)


def protoclr_ae(
        dataset,
        datapath,
        lr=1e-3,
        inner_lr=1e-3,
        gamma=1.0,
        distance="euclidean",
        ckpt_dir=Path("./ckpts"),
        ae=False,
        tau=1.0,
        eval_ways=5,
        clustering_alg=None,
        km_clusters=None,
        cl_reduction=None,
        cluster_on_latent=False,
        eval_support_shots=1,
        eval_query_shots=15,
        n_images=None,
        n_classes=None,
        n_support=1,
        n_query=3,
        batch_size=50,
        no_aug_support=False,
        no_aug_query=False,
        logging="wandb",
        log_images=False,
        profiler="torch",
        train_oracle_mode=False,
        train_oracle_ways=None,
        train_oracle_shots=None,
        num_workers=0,
        callbacks=True,
        patience=200,
        use_plotly=True,
        use_umap=True,
        use_pacmap=False,
        uuid=None,  # comes from OS should be constant mostly
):
    pl.seed_everything(42)
    gpus = torch.cuda.device_count()
    params = PCLRParamsContainer(
        dataset,
        datapath,
        seed=42,
        gpus=gpus,
        lr=lr,
        inner_lr=inner_lr,
        gamma=gamma,
        distance=distance,
        ckpt_dir=Path("./ckpts"),
        ae=ae,
        tau=tau,
        clustering_algo=clustering_alg,
        km_clusters=km_clusters,
        cl_reduction=cl_reduction,
        eval_ways=eval_ways,
        eval_support_shots=eval_support_shots,
        eval_query_shots=eval_query_shots,
        n_images=n_images,
        n_classes=n_classes,
        n_support=n_support,
        n_query=n_query,
        batch_size=batch_size,
        no_aug_support=no_aug_support,
        no_aug_query=no_aug_query,
        log_images=log_images,
        train_oracle_mode=train_oracle_mode,
        train_oracle_ways=train_oracle_ways,
        train_oracle_shots=train_oracle_shots,
        num_workers=num_workers,
        use_umap=use_umap,
        use_pacmap=use_pacmap
    )

    if (
            train_oracle_mode is True
            and train_oracle_ways is not None
            and train_oracle_shots is not None
    ):
        dm = OracleDataModule(params)
    else:
        dm = UnlabelledDataModule(params)

    model = ProtoCLR(params)

    cbs = []

    if logging == "wandb":
        logger = WandbLogger(
            project="ProtoCLR+AE",
            config={
                "batch_size": batch_size,
                "n_classes": n_classes,
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
                "no_aug_support": no_aug_support,
                "no_aug_query": no_aug_query,
                "clustering_algo": clustering_alg,
                "cl_reduction": cl_reduction,
                "clustering_on_latent": cluster_on_latent,
                "oracle_mode": train_oracle_mode,
                "train_oracle_ways": train_oracle_ways,
                "train_oracle_shots": train_oracle_shots,
                "umap": use_umap,
                "KM Clusters": km_clusters,
                "timestamp": str(datetime.now()),
            },
        )
        logger.watch(model)
        if callbacks:
            cbs = [
                EarlyStopping(monitor="val_loss", patience=300, min_delta=0.02),
                UMAPCallback(use_plotly=use_plotly),
                UMAPCallbackOnTrain(every_n_steps=50, use_plotly=use_plotly),
                PCACallback(use_plotly=use_plotly),
                PCACallbackOnTrain(every_n_steps=50, use_plotly=use_plotly),
            ]
            if ae:
                dataset_train = UnlabelledDataset(
                    dataset=dataset,
                    datapath=datapath,
                    split="train",
                    n_support=1,
                    n_query=3,
                )
                cbs.insert(0, WandbImageCallback(get_train_images(dataset_train, 8)))
        elif patience is not None:
            cbs = [
                EarlyStopping(monitor="val_accuracy", patience=patience, min_delta=0.02)
            ]
        # should be there no matter what?
        cbs.append(ConfidenceIntervalCallback())

    elif logging == "tb":
        logger = TensorBoardLogger(save_dir="tb_logs")
        dataset_train = UnlabelledDataset(
            dataset=dataset, datapath=datapath, split="train", n_support=1, n_query=3
        )
        cbs = (
            [TensorBoardImageCallback(get_train_images(dataset_train, 8))]
            if callbacks and ae
            else []
        )

    ckpt_path = os.path.join(
        ckpt_dir, f"{dataset}/{eval_ways}_{eval_support_shots}_om-{train_oracle_mode}/{str(datetime.now())}"
    )

    ckpt_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        dirpath=ckpt_path,
        filename="{epoch}-{step}-{val_loss:.2f}-{val_accuracy:.3f}",
        every_n_epochs=10,
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
        max_epochs=10000,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=15,
        limit_test_batches=600,
        callbacks=cbs,
        num_sanity_val_steps=1,
        weights_save_path=os.path.join(ckpt_dir, "hpc_saves", uuid),
        gpus=gpus,
        logger=logger,
        # strategy="dp"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.test(datamodule=dm, ckpt_path=ckpt_callback.best_model_path)
    wandb.finish()
