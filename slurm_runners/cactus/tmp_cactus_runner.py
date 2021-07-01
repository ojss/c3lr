import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import warnings
from torch import nn
from unsupervised_meta_learning.cactus import *
from unsupervised_meta_learning.protonets import ProtoModule, PrototypicalNetwork, CactusPrototypicalModel
from unsupervised_meta_learning.nn_utils import Flatten


def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        Flatten()
    )
    return encoder

dm = CactusDataModule(ways=20, shots=1, query=15, use_precomputed_partitions=False)
model = ProtoModule(encoder=CactusPrototypicalModel(in_channels=1, hidden_size=64), num_classes=20, lr=1e-3, cactus_flag=True)

logger = WandbLogger(
    project='protonet',
    config={
        'batch_size': 1,
        'steps': 30000,
        'dataset': "omniglot",
        'cactus': True,
        'pre-loaded-partitions': True,
        'partitions': 1
    }
)


trainer = pl.Trainer(
        profiler='simple',
#         max_steps=30_000,
        max_epochs=300,
        fast_dev_run=False,
        gpus=1,
        log_every_n_steps=25,
        check_val_every_n_epoch=1,
        flush_logs_every_n_steps=1,
        num_sanity_val_steps=2,
        logger=logger
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer.fit(model, datamodule=dm)

wandb.finish()
