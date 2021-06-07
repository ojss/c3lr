# AUTOGENERATED! DO NOT EDIT! File to edit: 01b_data_loaders_pl.ipynb (unless otherwise specified).

__all__ = ['OmniglotDataModule']

# Cell
#export

import warnings
from torchmeta.datasets.helpers import omniglot, miniimagenet, ClassSplitter
from torchmeta.datasets import Omniglot
from torchmeta.utils.data import BatchMetaDataLoader

import pytorch_lightning as pl

# Cell
class OmniglotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        shots: int,
        ways: int,
        shuffle_ds: bool,
        test_shots: int,
        meta_train: bool,
        download: bool,
        batch_size: str,
        shuffle: bool,
        num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.shots = shots
        self.ways = ways
        self.shuffle_ds = shuffle_ds
        self.test_shots = test_shots
        self.meta_train = meta_train
        self.download = download
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.task_dataset = omniglot(
            self.data_dir,
            shots=self.shots,
            ways=self.ways,
            shuffle=self.shuffle_ds,
            test_shots=self.test_shots,
            meta_train=self.meta_train,
            download=self.download
        )
    def train_dataloader(self):
        return BatchMetaDataLoader(
            self.task_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        self.val_tasks = omniglot(
            self.data_dir,
            shots=self.shots,
            ways=self.ways,
            shuffle=self.shuffle_ds,
            test_shots=self.test_shots,
            meta_val=True,
            download=self.download
        )
        return BatchMetaDataLoader(
            self.val_tasks,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        self.test_tasks = omniglot(
            self.data_dir,
            shots=self.shots,
            ways=self.ways,
            shuffle=self.shuffle_ds,
            test_shots=self.test_shots,
            meta_test=True,
            download=self.download
        )
        return BatchMetaDataLoader(
            self.test_tasks,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )