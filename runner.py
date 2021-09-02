import os
import argparse
import warnings

import fire
from pytorch_lightning import callbacks
import wandb
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from unsupervised_meta_learning.cactus import *
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataset, UnlabelledDataModule
from unsupervised_meta_learning.proto_utils import CAE, CAE4L
from unsupervised_meta_learning.protonets import ProtoModule, PrototypicalNetwork, CactusPrototypicalModel
from unsupervised_meta_learning.protoclr import ProtoCLR


def cactus(emb_data_dir:Path=None, n_ways=20, n_shots=1, query=15, batch_size=1, epochs=300, use_precomputed_partitions=False, final_chkpt_name='final.chkpt', final_chkpt_loc=os.getcwd()):

    dm = CactusDataModule(ways=n_ways, shots=n_shots, query=query, use_precomputed_partitions=use_precomputed_partitions, emb_data_dir=emb_data_dir)
    model = ProtoModule(encoder=CactusPrototypicalModel(in_channels=1, hidden_size=64), num_classes=20, lr=1e-3, cactus_flag=True)

    logger = WandbLogger(
        project='protonet',
        config={
            'batch_size': batch_size,
            'steps': 30000,
            'dataset': "omniglot",
            'cactus': True,
            'pre-loaded-partitions': use_precomputed_partitions,
            'partitions': 1,
            'n_ways': n_ways,
            'n_shots': n_shots,
            'query_shots': query
        },
        log_model=True
    )


    trainer = pl.Trainer(
            profiler='simple',
    #         max_steps=30_000,
            max_epochs=epochs,
            fast_dev_run=False,
            gpus=1,
            log_every_n_steps=25,
            check_val_every_n_epoch=1,
            flush_logs_every_n_steps=1,
            num_sanity_val_steps=2,
            logger=logger,
            default_root_dir='/home/nfs/oshirekar/unsupervised_ml/cactus_chkpnts/',
            checkpoint_callback=True
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(os.path.join(final_chkpt_loc, final_chkpt_name))

    wandb.finish()

    return 0


def protoclr_ae(dataset, datapath, eval_ways=5, eval_support_shots=1, eval_query_shots=15):
    dm = UnlabelledDataModule(dataset, datapath, split='train', transform=None,
                 n_support=1, n_query=3, n_images=None, n_classes=None, batch_size=50,
                 seed=10, mode='trainval', eval_ways=eval_ways, eval_support_shots=eval_support_shots, 
                 eval_query_shots=eval_query_shots)

    model = ProtoCLR(model=CAE4L(in_channels=1, out_channels=64, hidden_size=64), n_support=1, n_query=3, batch_size=50, lr_decay_step=25000, lr_decay_rate=.5, ae=True)

    logger = WandbLogger(
        project='ProtoCLR+AE',
        config={
            'batch_size': 50,
            'steps': 100,
            'dataset': dataset,
            'eval_ways': eval_ways,
            'eval_support_shots': eval_support_shots,
            'eval_query_shots': eval_query_shots
        }
    )
    trainer = pl.Trainer(
            profiler='simple',
            max_epochs=10000,
            limit_train_batches=100,
            fast_dev_run=False,
            limit_val_batches=15,
            limit_test_batches=600,
            callbacks=[EarlyStopping(monitor="val_loss", patience=300, min_delta=.02)],
            num_sanity_val_steps=2, gpus=1,
            logger=logger
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=dm)

    trainer.test()
    wandb.finish()

if __name__ == '__main__':
    fire.Fire({
        'cactus': cactus,
        'protoclr_ae': protoclr_ae
    })
