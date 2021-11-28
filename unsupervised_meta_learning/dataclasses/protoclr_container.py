from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from torch import nn

from unsupervised_meta_learning.proto_utils import Decoder4L, Decoder4L4Mini, Encoder4L


@dataclass
class PCLRParamsContainer:
    dataset: str
    datapath: str
    gpus: int = 0
    transform: Optional[list] = None
    n_support: int = 1
    n_query: int = 3
    n_images: Optional[int] = None
    n_classes: Optional[int] = None
    batch_size: int = 50
    no_aug_support: bool = True
    no_aug_query: bool = False
    seed: int = 42
    mode: str = "trainval"
    merge_train_val: bool = True
    num_workers: int = 2
    eval_ways: int = 5
    eval_support_shots: int = 5
    eval_query_shots: int = 15
    gamma: float = 1e-3
    lr: float = 1e-3
    inner_lr: float = 1e-3
    lr_decay_step: int = 25000
    lr_decay_rate: float = 0.5
    encoder_class: nn.Module = Encoder4L
    decoder_class: nn.Module = nn.Identity
    classifier: Optional[nn.Module] = None
    num_input_channels: int = 1
    base_channel_size: int = 64
    latent_dim: int = 64
    distance: str = "euclidean"
    ckpt_dir: Path = Path("./ckpts")
    tau: float = 1.0
    ae: bool = False
    clustering_algo: Optional[str] = None
    cl_reduction: str = "mean"
    log_images: bool = False
    sup_finetune: bool = True
    sup_finetune_lr: float = 1e-3
    sup_finetune_epochs: int = 15
    ft_freeze_backbone: bool = True
    finetune_batch_norm: bool = False
    train_oracle_mode: bool = False
    train_oracle_ways: Optional[int] = None
    train_oracle_shots: Optional[int] = None
    use_umap: bool = False

    def __post_init__(self):
        if self.dataset == "omniglot":
            self.num_input_channels = 1
            self.decoder_class = Decoder4L
        else:
            self.num_input_channels = 3
            self.decoder_class = Decoder4L4Mini
