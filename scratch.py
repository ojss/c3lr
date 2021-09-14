import warnings
from numpy import mod
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from unsupervised_meta_learning.pl_dataloaders import (UnlabelledDataModule,
                                                       UnlabelledDataset)
from unsupervised_meta_learning.proto_utils import CAE, CAE4L, Decoder4L, Encoder, Encoder4L, Decoder4L4Mini
from unsupervised_meta_learning.protoclr import (ConfidenceIntervalCallback,
                                                 ProtoCLR, WandbImageCallback,
                                                 get_train_images)

dm = UnlabelledDataModule('omniglot', './data/', split='train', transform=None,
                          n_support=1, n_query=3, n_images=None, n_classes=None, batch_size=50,
                          seed=10, mode='trainval', eval_ways=5, eval_support_shots=1,
                          eval_query_shots=15)

# net = CAE4L(in_channels=1, out_channels=64, hidden_size=64)
model = ProtoCLR(
    n_support=1, n_query=3, batch_size=50, distance='cosine', τ=.5,
    num_input_channels=1, decoder_class=Decoder4L,
    lr_decay_step=25000, lr_decay_rate=.5, ae=True, gamma=1., log_images=True)

# logger = WandbLogger(
#     project='ProtoCLR+AE',
#     config={
#         'batch_size': 50,
#         'steps': 100,
#         'dataset': "miniimagenet",
#         'testing': True
#     }
# )
dataset_train = UnlabelledDataset(
    dataset='omniglot',
    datapath='./data/',
    split='train',
    n_support=1,
    n_query=3
)

trainer = pl.Trainer(
    profiler='simple',
    max_epochs=1,
    limit_train_batches=100,
    fast_dev_run=False,
    limit_val_batches=15,
    limit_test_batches=600,
    callbacks=[EarlyStopping(monitor="val_loss", patience=200, min_delta=.02),
               ConfidenceIntervalCallback()],
    num_sanity_val_steps=2, gpus=1, #logger=logger
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
