import sys
import os
os.environ["SLURM_JOB_NAME"] = "bash"
del os.environ["SLURM_NTASKS"]
del os.environ["SLURM_JOB_NAME"]

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.tune import CLIReporter, analysis
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from torch.utils.data import dataset

from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.proto_utils import Decoder4L, AttnEncoder4L, Encoder4L
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule


def train_protoclr_ae(
    config, num_epochs=5000, num_gpus=0, dataset="omniglot", data_dir=None
):
    os.environ["SLURM_JOB_NAME"] = "bash"
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]

    model = ProtoCLR(**config)

    dm = UnlabelledDataModule(
        dataset,
        data_dir,
        transform=None,
        n_support=1,
        n_query=3,
        n_images=None,
        n_classes=None,
        batch_size=50,
        seed=10,
        mode="trainval",
        num_workers=2,
        eval_ways=5,
        eval_support_shots=5,
        eval_query_shots=15,
        train_oracle_mode=False,
    )

    metrics = {"val_loss": "val_loss", "val_acc": "val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=num_epochs,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=50,
        limit_test_batches=600,
        callbacks=[
            # UMAPCallback(every_n_epochs=1, use_plotly=False),
            # PCACallback(),
            # UMAPCallbackOnTrain(),
            # PCACallbackOnTrain()
            # UMAPClusteringCallback(f, cluster_alg="spectral", every_n_epochs=1, cluster_on_latent=True),
        ],
        num_sanity_val_steps=2,
        gpus=num_gpus,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
    )

    trainer.fit(model, datamodule=dm)


config = {
    "n_support": 1,
    "n_query": tune.choice([3, 4, 8]),
    "batch_size": 50,
    "distance": "euclidean",
    "τ": tune.choice([1.0, 0.5, 0.07]),
    "num_input_channels": 1,
    "decoder_class": Decoder4L,
    "encoder_class": Encoder4L,
    "lr": tune.loguniform(1e-4, 1e-2),
    "inner_lr": tune.loguniform(1e-4, 1e-2),
    "lr_decay_step": tune.choice([1000, 10000, 25000]),
    "lr_decay_rate": tune.loguniform(1e-2, 5e-1),
    "ae": tune.choice([True, False]),
    "gamma": tune.loguniform(1e-4, 1e-2),
    "clustering_algo": None,
    "oracle_mode": False,
    "use_entropy": False
}

os.environ["SLURM_JOB_NAME"] = "bash"
del os.environ["SLURM_NTASKS"]
del os.environ["SLURM_JOB_NAME"]

ray.init(address=os.environ["ip_head"])

print("Nodes in the Ray cluster:")
print(ray.nodes())

dataset = "omniglot"
num_epochs = 5000
data_dir = "/home/nfs/oshirekar/unsupervised_ml/data/"
gpus_per_trial = 1
cpus_per_task = sys.argv[1]

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

reporter = CLIReporter(metric_columns=["val_loss", "val_acc", "training_iteration"])
os.environ["SLURM_JOB_NAME"] = "bash"
del os.environ["SLURM_NTASKS"]
del os.environ["SLURM_JOB_NAME"]

analysis = tune.run(
    tune.with_parameters(
        train_protoclr_ae,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        dataset=dataset,
        data_dir=data_dir,
    ),
    resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    metric="val_acc",
    mode="max",
    config=config,
    num_samples=33,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_omni_asha",
)

