from os import EX_UNAVAILABLE
import fire
import pytorch_lightning as pl

from unsupervised_meta_learning.dataclasses.protoclr_container import (
    PCLRParamsContainer, )
from unsupervised_meta_learning.pl_dataloaders import (
    UnlabelledDataModule,
)
from unsupervised_meta_learning.protoclr import ProtoCLR
from unsupervised_meta_learning.callbacks.confinterval import ConfidenceIntervalCallback
from unsupervised_meta_learning import backbone
from unsupervised_meta_learning.cdfsl.datasets import EuroSAT_few_shot


# pl.seed_everything(42)


def evaluator(dataset, datapath, model_path: str = None, eval_ways=5, eval_support_shots=5, eval_query_shots=15,
              sup_finetune=False, ft_freeze_backbone=True):
    image_size = 224
    iter_num = 600
    few_shot_params = dict(n_way = eval_ways , n_support = eval_support_shots)

    params = PCLRParamsContainer(
        dataset,
        datapath,
        # seed=42,
        gpus=1,
        distance='euclidean',
        clustering_algo='hdbscan',
        km_clusters=5,
        encoder_class=backbone.ResNet10(),
        ae=False,
        cl_reduction='mean',
        eval_ways=eval_ways,
        eval_support_shots=eval_support_shots,
        eval_query_shots=eval_query_shots,
        no_aug_support=True,
        num_workers=6,
        use_umap=False,
        sup_finetune=sup_finetune,
        ft_freeze_backbone=ft_freeze_backbone,
        cdfsl_flg=True
    )
    model = ProtoCLR.load_from_checkpoint(model_path, params=params)
    trainer = pl.Trainer(gpus=1, limit_test_batches=600, callbacks=[ConfidenceIntervalCallback(plot=False)], precision=16)
    if dataset == 'eurosat':
        dmgr = EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = eval_query_shots, **few_shot_params)
    
    dl = dmgr.get_data_loader(aug=False)

    trainer.test(model=model, test_dataloaders=dl)


if __name__ == "__main__":
    fire.Fire(evaluator)
