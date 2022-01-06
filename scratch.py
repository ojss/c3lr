# -*- coding: utf-8 -*-
import warnings
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from unsupervised_meta_learning.callbacks.image_generation import get_train_images

from unsupervised_meta_learning.callbacks.umapcallbacks import *
from unsupervised_meta_learning.dataclasses.protoclr_container import PCLRParamsContainer
from unsupervised_meta_learning.pl_dataloaders import (
    OracleDataset,
    UnlabelledDataModule,
    OracleDataModule
)
from unsupervised_meta_learning.proto_utils import (
    Decoder4L,
    Decoder4L4Mini,
    Encoder4L,
)
from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.cdfsl.datasets import miniImageNet_few_shot
from unsupervised_meta_learning import backbone



pl.seed_everything(42)
gpus = torch.cuda.device_count()
train_oracle_mode = False
train_oracle_shots = None
train_oracle_ways = None

params = PCLRParamsContainer(
    "miniimagenet",
    "./data/untarred",
    gpus=gpus,
    transform=None,
    n_support=1,
    n_query=3,
    n_images=None,
    n_classes=None,
    batch_size=30,
    mode="trainval",
    num_workers=0,
    eval_ways=5,
    eval_support_shots=5,
    eval_query_shots=15,
    train_oracle_mode=train_oracle_mode,
    train_oracle_shots=train_oracle_shots,
    train_oracle_ways=train_oracle_ways,
    distance="euclidean",
    tau=0.5,
    num_input_channels=3,
    encoder_class=backbone.ResNet10(),
    lr_decay_step=25000,
    lr_decay_rate=0.5,
    clustering_algo='hdbscan',
    km_clusters=8,
    km_use_nearest=False,
    km_n_neighbours=30,
    cl_reduction="mean",
    ae=False,
    gamma=.001,
    log_images=True,
    use_umap=False,
    rerank_kjrd=True,

    cdfsl_flg=True,
    seed=42
)


datamgr = miniImageNet_few_shot.SimpleDataManager(224, batch_size = 30)
dm = datamgr.get_data_loader(aug = None,
                                          n_support=params.n_support, n_query=params.n_query,
                                          no_aug_support=params.no_aug_support, no_aug_query=params.no_aug_query)
# exit(0)
# if train_oracle_mode and train_oracle_shots is not None and train_oracle_ways is not None:
#     dm = OracleDataModule(params)
# else:
#     print("using unlabelled")
#     dm = UnlabelledDataModule(params)

model = ProtoCLR(params)

# logger = WandbLogger(
#     project="Scratch",
#     config={"batch_size": 100, "steps": 100, "dataset": "omniglot", "testing": True},
# )
# logger.watch(model)


# dataset_train = OracleDataset(
#                     dataset='miniimagenet',
#                     datapath='./data/untarred/',
#                     split="train",
#                     n_support=1,
#                     n_query=3,
#                     no_aug_support=True,
#                     train_oracle_mode=False,
#                     train_oracle_shots=5,
#                     train_oracle_ways=5
#                 )

trainer = pl.Trainer(
    # profiler="simple",
    max_epochs=1,
    limit_train_batches=2,
    fast_dev_run=True,
    limit_val_batches=0,
    limit_test_batches=1,
    val_check_interval=0,
    callbacks=[
        # UMAPConstantInput(input_images=get_train_images(dataset_train, 50))
    ],
    num_sanity_val_steps=0,
    gpus=1,
    # logger=logger
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer.fit(model, train_dataloader=dm)

# trainer.test(model=model, datamodule=dm)

# from torchinfo import summary
# import torch.nn as nn

# net = Decoder4L4Mini(out_channels=64, mode='nearest')
# batch_size = 50
# summary(model, input_size=(batch_size, 3, 84, 84))
