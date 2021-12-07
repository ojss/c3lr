__all__ = ['task_function']
from dataclasses import asdict
from datetime import datetime
import pytorch_lightning as pl
import torch as tr
from hydra_zen import (builds, instantiate, just, make_config,
                       make_custom_builds_fn)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from unsupervised_meta_learning.dataclasses.protoclr_container import \
    PCLRParamsContainer
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule
from unsupervised_meta_learning.proto_utils import (AttnEncoder4L, Decoder4L,
                                                    Decoder4L4Mini, Encoder4L)
from unsupervised_meta_learning.protoclr import ProtoCLR

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

EncConf = builds(Encoder4L, zen_partial=True, hydra_recursive=True)
DecConf = builds(Decoder4L4Mini, zen_partial=True, hydra_recursive=True)

ParamsConf = builds(
    PCLRParamsContainer,
    "miniimagenet",
    "/home/nfs/oshirekar/unsupervised_ml/data/",
    transform=None,
    n_support=1,
    n_query=3,
    no_aug_support=True,
    no_aug_query=False,
    batch_size=200,
    seed=10,
    mode="trainval",
    num_workers=6,
    eval_ways=5,
    eval_support_shots=5,
    eval_query_shots=15,
    gamma=1e-2,
    lr=1e-3,
    inner_lr=1e-3,
    lr_decay_step=25000,
    lr_decay_rate=0.5,
    num_input_channels=3,
    distance="euclidean",
    tau=1.0,
    ae=False,
    # Clustering parameters
    clustering_algo='hdbscan',
    km_clusters=5,  # !IMP: doubles as the parameter for hdb_min_cluster_size TODO: correct the name
    km_use_nearest=False,
    km_n_neighbours=50,
    cl_reduction="mean",

    log_images=False,
    train_oracle_mode=False,

    # UMAP settings
    use_umap=False,
    umap_min_dist=.25,
    rdim_n_neighbors=50,
    rdim_components=2,

    # ReRanker opts
    rerank_kjrd=True,

    ckpt_dir="./ckpts",
    zen_partial=True,
    populate_full_signature=True,
)

WandbConf = pbuilds(WandbLogger, project="ProtoCLR-C-Sweeps", group="hydra-umap-sweeps", hydra_recursive=True)

DataConf = pbuilds(UnlabelledDataModule)
LitConf = pbuilds(ProtoCLR, params=ParamsConf, hydra_recursive=True)
EStopConf = builds(EarlyStopping, monitor='val_accuracy', patience=200)
MCkpt = builds(
        ModelCheckpoint,
        monitor="val_accuracy",
        mode="max",
        dirpath=f'/home/nfs/oshirekar/unsupervised_ml/ckpts/sweeps/umap/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
        filename="{epoch}-{step}-{val_loss:.2f}-{val_accuracy:.3f}",
        every_n_epochs=5,
        save_top_k=20)

if tr.cuda.is_available():
    gpus = -1
else:
    gpus = 0

TrainerConf = pbuilds(
    pl.Trainer,
    profiler="simple",
    max_epochs=10000,
    limit_train_batches=100,
    fast_dev_run=False,
    limit_val_batches=15,
    limit_test_batches=600,
    callbacks=[EStopConf, MCkpt],
    num_sanity_val_steps=1,
    # weights_save_path=os.path.join(ckpt_dir, "hpc_saves", uuid),
    logger=WandbConf,
    gpus=gpus,
    hydra_recursive=True
)

ExperimentConfig = make_config(
    enc=EncConf,
    dec=DecConf,
    params=ParamsConf,
    estop=EStopConf,
    mckpt=MCkpt,
    trainer=TrainerConf,
    datamodule=DataConf,
    lit_module=LitConf,
    wb=WandbConf,
    hydra_recursive=True
)


def task_function(cfg: ExperimentConfig):
    print(cfg)
    from sklearnex import patch_sklearn
    patch_sklearn()

    obj = instantiate(cfg)
    params = obj.params(encoder_class=Encoder4L, decoder_class=Decoder4L4Mini)
    lit_module = obj.lit_module(params=params)
    datamodule = obj.datamodule(params=params)
    logger = obj.wb(config={'umap_rdim': params.rdim_components})
    trainer = obj.trainer()
    trainer.logger = logger
    trainer.fit(lit_module, datamodule=datamodule)
    val_acc = trainer.callback_metrics.get("val_accuracy")
    trainer.test(datamodule=datamodule, ckpt_path=obj.mckpt.best_model_path)
    test_acc = trainer.callback_metrics.get("test_accuracy")
    return val_acc
    # try:
    #     os.environ["SLURM_JOB_NAME"] = "bash"
    #     del os.environ["SLURM_NTASKS"]
    #     del os.environ["SLURM_JOB_NAME"]
    # except Exception as e:
    #     pass
