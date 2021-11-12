import sys
from pathlib import Path
import os
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
from ray.tune.integration.wandb import WandbLoggerCallback
from torch.utils.data import dataset

from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.proto_utils import Decoder4L, AttnEncoder4L, Encoder4L
from unsupervised_meta_learning.pl_dataloaders import UnlabelledDataModule


def train_protoclr_ae(
    config, num_epochs=5000, num_gpus=0, dataset="omniglot", data_dir=None
):
    try:
        os.environ["SLURM_JOB_NAME"] = "bash"
        del os.environ["SLURM_NTASKS"]
        del os.environ["SLURM_JOB_NAME"]
    except Exception as e:
        pass

    ckpt_dir = Path("./ckpts")
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
    callbacks = [
        TuneReportCheckpointCallback(
            metrics, on="validation_end", filename="tuner_ckpt"
        )
    ]

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=num_epochs,
        limit_train_batches=100,
        fast_dev_run=False,
        limit_val_batches=50,
        limit_test_batches=600,
        callbacks=[callbacks],
        gpus=num_gpus,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        resume_from_checkpoint=os.path.join(ckpt_dir, "tuner_ckpt"),
    )

    trainer.fit(model, datamodule=dm)


config = {
    "n_support": 1,
    "n_query": tune.choice([3, 4, 8]),
    "batch_size": 50,
    "distance": "euclidean",
    "τ": 1.0,
    "num_input_channels": 1,
    "decoder_class": Decoder4L,
    "encoder_class": Encoder4L,
    "lr": 1e-3,
    "inner_lr": 1e-3,
    "lr_decay_step": tune.choice([1000, 10000, 25000]),
    "lr_decay_rate": 5e-1,
    "ae": tune.choice([True, False]),
    "gamma": 1e-3,
    "clustering_algo": None,
    "oracle_mode": False,
    "use_entropy": False,
}

ray.init(address=os.environ["ip_head"])

print("Nodes in the Ray cluster:")
print(ray.nodes())

dataset = "omniglot"
num_epochs = 5000
data_dir = "/home/nfs/oshirekar/unsupervised_ml/data/"
gpus_per_trial = 1
cpus_per_task = sys.argv[1]

# scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

scheduler = PopulationBasedTraining(
    perturbation_interval=50,
    hyperparam_mutations={
        "gamma": tune.loguniform(1e-4, 1e-2),
        "τ": tune.loguniform(0.5, 1.0),
        "lr": tune.loguniform(1e-4, 1e-2),
        "inner_lr": tune.loguniform(1e-4, 1e-2),
        "lr_decay_rate": tune.loguniform(1e-2, 5e-1),
    },
)

reporter = CLIReporter(metric_columns=["val_loss", "val_acc", "training_iteration"])

analysis = tune.run(
    tune.with_parameters(
        train_protoclr_ae,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        dataset=dataset,
        data_dir=data_dir,
    ),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    metric="val_acc",
    mode="max",
    config=config,
    num_samples=33,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_omni_pbt",
    callbacks=[
        WandbLoggerCallback(
            project="ProtoCLR+AE",
            api_key="f586c47014afd7c57efe3f28e80e1597b57dddd1",
            log_config=True,
        )
    ],
)

print("Best hyperparameters found were: ", analysis.best_config)
