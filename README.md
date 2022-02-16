 <!-- TODO: Superscript C3LR -->
# This Repo holds the code for C<sup>3</sup>LR

# Introduction
Unsupervised learning is argued to be the dark matter of human intelligence. To build in this direction, this paper focuses on unsupervised learning from an abundance of unlabeled data followed by few-shot fine-tuning on a downstream classification task. To this aim, we extend a recent study on adopting contrastive learning for unsupervised pre-training by incorporating class-level cognizance and expanding the contrastive optimization loss to account for it. Our experimentation both in standard and cross-domain scenarios demonstrate that we not only stay current with the state-of-the-art (SoTA) in all scenarios but also sets a new SoTA in (5-way, 1 and 5-shot) settings for mini-ImageNet dataset.

# Algorithm
Algorithm Flow             |  Algorithm
:-------------------------:|:-------------------------:
![](images/flow.png "C3LR")  |  ![](images/algo.png)

The algorithm consists of the following major high-level components:

1. Batch generation (including augmentation)
2. Re-ranking and Clustering
3. Class-cognizant contrastive loss
4. Standard contrastive loss


# Results on Omniglot and mini-ImageNet

| Method(N,K) | (5,1) | (5,5) | (5,1) | (5,5) |
|--------------|-------|-------|-------|-------|
| CACTUs-MAML                               | 68.84 ± 0.80             | 87.78 ± 0.50             | 39.90 	± 0.74                    | 53.97 ± 0.70             |
| CACTUs-ProtoNet                           | 68.12 ± 0.84             | 83.58 ± 0.61             | 39.18 ± 0.71             | 53.36 ± 0.70             |
| UMTRA                                     | 83.80                                     | 95.43                                     | 39.93                                     | 50.73                                     |
| AAL-ProtoNet                              | 84.66 ± 0.70             | 89.14 ± 0.27             | 37.67 ± 0.39             | 40.29 ± 0.68             |
| AAL-MAML++                                | 88.40 ± 0.75             | 97.96 ± 0.32 | 34.57 ± 0.74             | 49.18± 0.47              |
| UFLST                                     | 97.03                            | 99.19                            | 33.77 ± 0.70             | 45.03 ± 0.73             |
| ULDA-ProtoNet                             | -                                         | -                                         | 40.63 ± 0.61             | 55.41 ± 0.57             |
| ULDA-MetaOptNet                           | -                                         | -                                         | 40.71 ± 0.62             | 54.49 ± 0.58             |
| U-SoSN+ ArL                               | -                                         | -                                         | 41.13 ± 0.84             | 55.39 ± 0.79             |
| LASIUM                                    | 83.26 ± 0.55             | 95.29 ± 0.22             | 40.19 ± 0.58             | 54.56 ± 0.55             |
| ProtoTransfer (L=50)     | 88.00 ± 0.64             | 96.48 ± 0.26             | 45.67 ± 0.79 | 62.99 ± 0.75 |
| ProtoTransfer (L=200)      | 88.37 ± 0.74             | 96.54 ± 0.41             | 44.17 ± 1.08             | 61.07 ± 0.82             |
| **C<sup>3</sup>LR** (**ours**) | 89.30 ± 0.64 | 97.38 ± 0.23             | 47.92 ± 1.2     | 64.81 ± 1.15    |
| MAML  (supervised)                 | 94.46 ± 0.35 | 98.83 ± 0.12 | 46.81 ± 0.77 | 62.13 ± 0.72  |
| ProtoNet  (supervised)              | 97.70 ± 0.29  | 99.28 ± 0.10 | 46.44 ± 0.78  | 66.33 ± 0.68  |
| MMC   (supervised)                                    | 97.68 ± 0.07  | -                             | 50.41 ± 0.31 | 64.39 ± 0.24 |
| FEAT  (supervised)                                      | -                             | -                             | 55.15                         | 71.61                         |
| Pre+Linear   (supervised) | 94.30 ± 0.43 | 99.08 ± 0.10 | 43.87 ± 0.69 | 63.01 ± 0.71 |

