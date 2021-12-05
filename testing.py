# %%
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
    PCLRParamsContainer, ReRankerContainer,
)

# %%
# pl.seed_everything(42)

# %%
dataset='miniimagenet'
datapath='./data/untarred'
lr=1e-3
inner_lr=1e-3
gamma=1.0
distance="euclidean"
ckpt_dir=Path("./ckpts")
ae=False
tau=1.0
eval_ways=5
clustering_alg='kmeans'
clustering_callback=False
km_clusters=25
km_use_nearest=True
km_n_neighbours=50
cl_reduction='mean'
eval_support_shots=20
eval_query_shots=15
n_images=None
n_classes=None
n_support=1
n_query=3
batch_size=200
no_aug_support=True
no_aug_query=False
logging="wandb"
log_images=False
profiler="torch"
train_oracle_mode=False
train_oracle_ways=None
train_oracle_shots=None
num_workers=3
callbacks=True
patience=200
use_plotly=True
use_umap=False
umap_min_dist: float = .25
rdim_n_neighbors: int = 50
rdim_components: int = 2
rerank_kjrd=False
rrk1=20
rrk2=6
rrlambda=0
uuid=None

# %%
params = PCLRParamsContainer(
        dataset,
        datapath,
        seed=42,
        gpus=1,
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

# %%
model = ProtoCLR(params)

# %%
model = ProtoCLR.load_from_checkpoint("./data/copper-durain-27-best-epoch-184/epoch=144-step=14499-val_loss=1.50-val_accuracy=0.571.ckpt", params=params)

# %%
trainer = pl.Trainer(gpus=1, limit_test_batches=600)

# %%
dm = UnlabelledDataModule(params)

# %%
trainer.test(model=model, datamodule=dm)

# %%



