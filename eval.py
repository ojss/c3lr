import fire
import pytorch_lightning as pl

from unsupervised_meta_learning.dataclasses.protoclr_container import (
    PCLRParamsContainer, )
from unsupervised_meta_learning.pl_dataloaders import (
    UnlabelledDataModule,
)
from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.callbacks.confinterval import ConfidenceIntervalCallback


# pl.seed_everything(42)


def evaluator(dataset, datapath, model_path: str = None, eval_ways=5, eval_support_shots=5, eval_query_shots=15,
              sup_finetune=False, ft_freeze_backbone=True):
    params = PCLRParamsContainer(
        dataset,
        datapath,
        # seed=42,
        gpus=1,
        distance='euclidean',
        clustering_algo='hdbscan',
        km_clusters=5,
        cl_reduction='mean',
        eval_ways=eval_ways,
        eval_support_shots=eval_support_shots,
        eval_query_shots=eval_query_shots,
        no_aug_support=True,
        num_workers=2,
        use_umap=False,
        sup_finetune=sup_finetune,
        ft_freeze_backbone=ft_freeze_backbone
    )
    model = ProtoCLR.load_from_checkpoint(model_path, params=params)
    trainer = pl.Trainer(gpus=1, limit_test_batches=600, callbacks=[ConfidenceIntervalCallback(plot=False)])
    dm = UnlabelledDataModule(params)
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    fire.Fire(evaluator)
