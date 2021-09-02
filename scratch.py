import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from unsupervised_meta_learning.proto_utils import prototypical_loss, get_prototypes, CAE, CAE4L
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule, UnlabelledDataset
from unsupervised_meta_learning.protoclr import ProtoCLR


dm = UnlabelledDataModule('omniglot', './data/untarred/', split='train', transform=None,
                 n_support=1, n_query=3, n_images=None, n_classes=None, batch_size=50,
                 seed=10, mode='trainval', eval_ways=5, eval_support_shots=1, 
                 eval_query_shots=15)

model = ProtoCLR(model=CAE4L(in_channels=1, out_channels=64, hidden_size=64),
 n_support=1, n_query=3, batch_size=50, lr_decay_step=25000, lr_decay_rate=.5, ae=True)

logger = WandbLogger(
    project='ProtoCLR+AE',
    config={
        'batch_size': 50,
        'steps': 100,
        'dataset': "omniglot"
    }
)
trainer = pl.Trainer(
        profiler='simple',
        max_epochs=1,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=15,
        limit_test_batches=600,
        callbacks=[EarlyStopping(monitor="val_loss", patience=200, min_delta=.02)],
        num_sanity_val_steps=2, gpus=1, #logger=logger
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer.fit(model, datamodule=dm)

trainer.test()
