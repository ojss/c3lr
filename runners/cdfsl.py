import os
import wandb
import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from unsupervised_meta_learning import backbone, callbacks
from unsupervised_meta_learning.cdfsl.datasets import miniImageNet_few_shot
from unsupervised_meta_learning.dataclasses.protoclr_container import (
    PCLRParamsContainer,
    ReRankerContainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from unsupervised_meta_learning.protoclr import ProtoCLR


def cdfsl_train(
    dataset="miniImageNet",
    max_epochs=400,
    lr=1e-3,
    inner_lr=1e-3,
    distance="euclidean",
    ckpt_dir=Path("./ckpts"),
    tau=1.0,
    eval_ways=5,
    clustering_alg='hdbscan',
    km_clusters=5,
    km_use_nearest=False,
    km_n_neighbours=30,
    cl_reduction='mean',
    eval_support_shots=1,
    eval_query_shots=15,
    n_images=None,
    n_classes=None,
    n_support=1,
    n_query=3,
    batch_size=50,
    no_aug_support=True,
    no_aug_query=False,
    logging="wandb",
    log_images=False,
    num_workers=0,
    use_umap=False,
    rerank_kjrd=True,
    rrk1=20,
    rrk2=6,
    rrlambda=0,
):
    pl.seed_everything(42)
    gpus = torch.cuda.device_count()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    params = PCLRParamsContainer(
        "miniimagenet",
        None,
        tau=tau,
        gpus=gpus,
        transform=None,
        n_support=n_support,
        n_query=n_query,
        n_images=n_images,
        n_classes=n_classes,
        batch_size=batch_size,
        merge_train_val=False,
        num_workers=0,
        eval_ways=eval_ways,
        eval_support_shots=eval_support_shots,
        eval_query_shots=eval_query_shots,
        train_oracle_mode=False,
        distance="euclidean",
        num_input_channels=3,
        encoder_class=backbone.ResNet10(),
        lr_decay_step=25000,
        lr_decay_rate=0.5,
        clustering_algo='hdbscan',
        km_clusters=km_clusters,
        cl_reduction=cl_reduction,

        log_images=False,
        use_umap=False,
        rerank_kjrd=True,

        cdfsl_flg=True,
        seed=42
    )


    if dataset == "miniImageNet":
        datamgr = miniImageNet_few_shot.SimpleDataManager(224, batch_size=batch_size)
    train_dataloader = datamgr.get_data_loader(
        aug=None,
        n_support=params.n_support,
        n_query=params.n_query,
        no_aug_support=params.no_aug_support,
        no_aug_query=params.no_aug_query,
    )
    model = ProtoCLR(params)

    if logging == "wandb":
        logger = WandbLogger(
            project="ProtoCLR-C-CDFSL",
            log_model=True,
            config={
                "SLURM_JOB_ID": os.environ["SLURM_JOB_ID"],
                "batch_size": batch_size,
                "n_classes": n_classes,
                "lr": lr,
                "inner_lr": inner_lr,
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
                "KM Clusters": km_clusters,
                "Re-Rank": rerank_kjrd,
                "RRK1": rrk1,
                "RRK2": rrk2,
                "RRLambda": rrlambda,
                "timestamp": timestamp,
            },
        )
        logger.watch(model)

    elif logging == "tb":
        logger = TensorBoardLogger(save_dir="tb_logs")

    ckpt_path = os.path.join(
        ckpt_dir, f"cdfsl/train/miniImageNet/{timestamp}"
    )
    ckpt_callback = ModelCheckpoint(
        monitor='train_accuracy_epoch',  # previously val_accuracy
        mode="max",
        dirpath=ckpt_path,
        filename="{epoch}-{step}-{loss_epoch:.2f}-{train_accuracy_epoch:.3f}",
        every_n_epochs=1,
        save_top_k=5,
        save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        fast_dev_run=False,
        limit_val_batches=0,
        # val_check_interval=0,
        num_sanity_val_steps=0,
        gpus=gpus,
        callbacks=[ckpt_callback],
        logger=logger
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, train_dataloader=train_dataloader)
    
    print(f"Best Model Path: {ckpt_callback.best_model_path}")
    wandb.finish()
