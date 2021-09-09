import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from unsupervised_meta_learning.proto_utils import CAE, CAE4L
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule, UnlabelledDataset
from unsupervised_meta_learning.protoclr import ProtoCLR, WandbImageCallback, get_train_images


dm = UnlabelledDataModule('omniglot', './data/untarred/', split='train', transform=None,
                 n_support=1, n_query=3, n_images=None, n_classes=None, batch_size=50,
                 seed=10, mode='trainval', eval_ways=5, eval_support_shots=1, 
                 eval_query_shots=15)

# net = CAE4L(in_channels=1, out_channels=64, hidden_size=64)
model = ProtoCLR(n_support=1, n_query=3, batch_size=50, lr_decay_step=25000, lr_decay_rate=.5, ae=True, gamma=1., log_images=True)

logger = WandbLogger(
    project='ProtoCLR+AE',
    config={
        'batch_size': 50,
        'steps': 100,
        'dataset': "omniglot",
        'testing': True
    }
)
dataset_train = UnlabelledDataset(
    dataset='omniglot',
    datapath='./data/untarred/',
    split='train',
    n_support=1,
    n_query=3
)

trainer = pl.Trainer(
        profiler='simple',
        max_epochs=30,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=15,
        limit_test_batches=600,
        callbacks=[WandbImageCallback(get_train_images(dataset_train, 8), every_n_epochs=5),
        EarlyStopping(monitor="val_loss", patience=200, min_delta=.02)],
        num_sanity_val_steps=2, gpus=1, logger=logger
    )

logger.watch(model)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer.fit(model, datamodule=dm)

trainer.test()
