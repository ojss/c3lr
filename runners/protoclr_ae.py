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
    get_episode_loader
)
from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.dataclasses.protoclr_container import (
    PCLRParamsContainer, ReRankerContainer,
)


def protoclr_ae(
        dataset,
        datapath,
        merge_train_val=False,
        lr=1e-3,
        inner_lr=1e-3,
        gamma=1.0,
        distance="euclidean",
        ckpt_dir=Path("./ckpts"),
        ae=False,
        tau=1.0,
        eval_ways=5,
        clustering_alg=None,
        clustering_callback=False,
        km_clusters=None,
        km_use_nearest=False,
        km_n_neighbours=30,
        cl_reduction=None,
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
        umap_min_dist: float = .25,
        rdim_n_neighbors: int = 50,
        rdim_components: int = 2,
        rerank_kjrd=False,
        rrk1=20,
        rrk2=6,
        rrlambda=0,
        estop_ckpt_on_val_acc=False,
        uuid=None  # comes from OS should be constant mostly
):
    cluster_on_latent = False if use_umap else True
    if estop_ckpt_on_val_acc:
        monitor = "val_accuracy"
        ckpt_file_format = "{epoch}-{step}-{val_loss:.2f}-{val_accuracy:.3f}"
    else:
        monitor = "train_accuracy_epoch"
        ckpt_file_format = "{epoch}-{step}-{loss_epoch:.2f}-{train_accuracy_epoch:.3f}"

    pl.seed_everything(42)
    gpus = torch.cuda.device_count()

    params = PCLRParamsContainer(
        dataset,
        datapath,
        merge_train_val=merge_train_val,
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
        km_use_nearest=km_use_nearest,
        km_n_neighbours=km_n_neighbours,
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
        umap_min_dist=umap_min_dist,
        rdim_components=int(rdim_components),
        rdim_n_neighbors=int(rdim_n_neighbors),
        rerank_kjrd=rerank_kjrd,
        re_rank_args=ReRankerContainer(k1=rrk1, k2=rrk2, lambda_value=rrlambda)
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
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if logging == "wandb":
        logger = WandbLogger(
            project="ProtoCLR-C",
            config={
                "SLURM_JOB_ID": os.environ['SLURM_JOB_ID'],
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
                # TODO remove complex logic like this
                "oracle_mode": train_oracle_mode if train_oracle_mode and clustering_alg is None else False,
                "train_oracle_ways": train_oracle_ways,
                "train_oracle_shots": train_oracle_shots,
                "umap": use_umap,
                "KM Clusters": km_clusters,
                "Re-Rank": rerank_kjrd,
                "RRK1": rrk1,
                "RRK2": rrk2,
                "RRLambda": rrlambda,
                "estop_ckpt_on_val_acc": estop_ckpt_on_val_acc,
                "timestamp": timestamp
            },
        )
        logger.watch(model)
        if callbacks:
            cbs.extend([
                UMAPCallback(use_plotly=use_plotly),
                UMAPCallbackOnTrain(every_n_steps=50, use_plotly=use_plotly),
                PCACallback(use_plotly=use_plotly),
                PCACallbackOnTrain(every_n_steps=50, use_plotly=use_plotly),
            ])
            if ae:
                dataset_train = UnlabelledDataset(
                    dataset=dataset,
                    datapath=datapath,
                    split="train",
                    n_support=1,
                    n_query=3,
                )
                cbs.append(WandbImageCallback(get_train_images(dataset_train, 8)))
        if patience is not None:
            cbs.insert(0, EarlyStopping(monitor=monitor, patience=patience, mode='max'))
        if clustering_callback:
            tmp_dl = get_episode_loader(dataset, datapath, 5, 30, 15, 1, 'train')
            xs = next(iter(tmp_dl))
            cbs.append(
                UMAPConstantInput(
                    input_images=xs
                )
            )
            del tmp_dl
        # should be there no matter what?
        cbs.append(ConfidenceIntervalCallback())

    elif logging == "tb":
        logger = TensorBoardLogger(save_dir="tb_logs")
        dataset_train = UnlabelledDataset(
            dataset=dataset, datapath=datapath, split="train", n_support=1, n_query=3
        )
        cbs.extend(
            [TensorBoardImageCallback(get_train_images(dataset_train, 8))]
            if callbacks and ae
            else []
        )

    ckpt_path = os.path.join(
        ckpt_dir, f"{dataset}/{eval_ways}_{eval_support_shots}_om-{train_oracle_mode}/{timestamp}"
    )

    ckpt_callback = ModelCheckpoint(
        monitor=monitor,  # previously val_accuracy
        mode="max",
        dirpath=ckpt_path,
        filename=ckpt_file_format,
        every_n_epochs=1,
        save_top_k=20,
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
        fast_dev_run=True,
        limit_val_batches=15,
        limit_test_batches=600,
        callbacks=cbs,
        num_sanity_val_steps=1,
        # weights_save_path=os.path.join(ckpt_dir, "hpc_saves", uuid),
        gpus=gpus,
        logger=logger,
        # strategy="dp"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.test(datamodule=dm, ckpt_path=ckpt_callback.best_model_path)
    wandb.finish()
