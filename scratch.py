import warnings
from numpy import mod
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from unsupervised_meta_learning.pl_dataloaders import (
    UnlabelledDataModule,
    get_episode_loader,
    UnlabelledDataset,
)
from unsupervised_meta_learning.proto_utils import (
    CAE,
    CAE4L,
    Decoder4L,
    Encoder,
    Decoder,
    Encoder4L,
    Decoder4L4Mini,
    get_images_labels_from_dl,
)
from unsupervised_meta_learning.protoclr import (
    ConfidenceIntervalCallback,
    ProtoCLR,
    WandbImageCallback,
    UMAPCallback,
    UMAPClusteringCallback,
    get_train_images,
)

dm = UnlabelledDataModule(
    "omniglot",
    "./data/untarred",
    split="train",
    transform=None,
    n_support=1,
    n_query=3,
    n_images=None,
    n_classes=None,
    batch_size=50,
    seed=10,
    mode="trainval",
    num_workers=0,
    eval_ways=5,
    eval_support_shots=1,
    eval_query_shots=15,
)

model = ProtoCLR(
    n_support=1,
    n_query=3,
    batch_size=50,
    distance="euclidean",
    τ=0.5,
    num_input_channels=1,
    decoder_class=Decoder4L,
    encoder_class=Encoder4L,
    lr_decay_step=25000,
    lr_decay_rate=0.5,
    ae=True,
    gamma=.001,
    log_images=True,
    clustering_algo="kmeans",
)

# logger = WandbLogger(
#     project="ProtoCLR+AE",
#     config={"batch_size": 50, "steps": 100, "dataset": "omniglot", "testing": True},
# )
dataset_train = UnlabelledDataset(
    dataset="omniglot",
    datapath="./data/untarred",
    split="train",
    n_support=1,
    n_query=3,
)

dl = get_episode_loader(
    "omniglot",
    "./data/untarred",
    ways=5,
    shots=5,
    test_shots=15,
    batch_size=1,
    split="val",
    num_workers=0,
)

f = partial(get_images_labels_from_dl, dl)

trainer = pl.Trainer(
    profiler="simple",
    max_epochs=2,
    limit_train_batches=100,
    fast_dev_run=False,
    limit_val_batches=15,
    limit_test_batches=600,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=200, min_delta=0.02),
        # UMAPCallback(f, every_n_epochs=1)
        # UMAPClusteringCallback(f, cluster_alg="spectral", every_n_epochs=1, cluster_on_latent=True),
    ],
    num_sanity_val_steps=2,
    gpus=1,
    # logger=logger,
)

# logger.watch(model)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer.fit(model, datamodule=dm)

trainer.test()

# from torchinfo import summary
# import torch.nn as nn

# net = Decoder4L4Mini(out_channels=64, mode='nearest')
# batch_size = 50
# summary(model, input_size=(batch_size, 3, 84, 84))
