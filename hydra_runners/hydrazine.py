__all__ = ['task_function']
from pytorch_lightning.callbacks import EarlyStopping

import torch as tr
from torch.optim import Adam
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from hydra_zen import builds, just, make_config, make_custom_builds_fn, instantiate

from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.proto_utils import (
    Decoder4L,
    AttnEncoder4L,
    Decoder4L4Mini,
    Encoder4L,
)
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule
from unsupervised_meta_learning.dataclasses.protoclr_container import (
    PCLRParamsContainer,
)

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

EncConf = builds(Encoder4L, zen_partial=True, hydra_recursive=True)
DecConf = builds(Decoder4L4Mini, zen_partial=True, hydra_recursive=True)

ParamsConf = builds(
    PCLRParamsContainer,
    "miniimagenet",
    "/home/ojas/projects/unsupervised-meta-learning/data/untarred",
    transform=None,
    n_support=1,
    n_query=3,
    no_aug_support=True,
    no_aug_query=False,
    batch_size=50,
    seed=10,
    mode="trainval",
    num_workers=2,
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
    clustering_algo='hdbscan'
    km_clusters=5  # !IMP: doubles as the parameter for hdb_min_cluster_size TODO: correct the name
    km_use_neares=False
    km_n_neighbours=50
    cl_reduction="mean"

    log_images=False,
    train_oracle_mode=False,
    train_oracle_ways=10,
    train_oracle_shots=20,

    # UMAP settings
    use_umap=False
    umap_min_dist=.25
    rdim_n_neighbors=50
    rdim_components=2

    # ReRanker opts
    rerank_kjrd=True

    ckpt_dir="./ckpts",
    zen_partial=True,
    populate_full_signature=True,
)
DataConf = pbuilds(UnlabelledDataModule)
LitConf = pbuilds(ProtoCLR, params=ParamsConf, hydra_recursive=True)
EStopConf = builds(EarlyStopping, monitor='val_accuracy', patience=200)


if tr.cuda.is_available():
    gpus = -1
else:
    gpus = 0

TrainerConf = builds(
    pl.Trainer,
    profiler="simple",
    max_epochs=10000,
    limit_train_batches=100,
    fast_dev_run=False,
    limit_val_batches=15,
    limit_test_batches=600,
    callbacks=[EStopConf],
    num_sanity_val_steps=1,
    # weights_save_path=os.path.join(ckpt_dir, "hpc_saves", uuid),
    gpus=gpus,
)

ExperimentConfig = make_config(
    enc=EncConf,
    dec=DecConf,
    params=ParamsConf,
    estop=EStopConf,
    trainer=TrainerConf,
    datamodule=DataConf,
    lit_module=LitConf,
    hydra_recursive=True
)


def task_function(cfg: ExperimentConfig):
    print(cfg)
    obj = instantiate(cfg)
    params = obj.params(encoder_class=Encoder4L, decoder_class=Decoder4L4Mini)
    lit_module = obj.lit_module(params=params)
    datamodule = obj.datamodule(params=params)
    obj.trainer.fit(lit_module, datamodule=datamodule)
    # obj.trainer.test(datamodule=datamodule)
    val_acc = obj.trainer.callback_metrics.get("val_accuracy")
    # test_acc = obj.trainer.callback_metrics.get("test_accuracy")
    return val_acc
    # try:
    #     os.environ["SLURM_JOB_NAME"] = "bash"
    #     del os.environ["SLURM_NTASKS"]
    #     del os.environ["SLURM_JOB_NAME"]
    # except Exception as e:
    #     pass
