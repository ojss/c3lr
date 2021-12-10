#!/bin/bash

#SBATCH --gres=gpu:1
# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=medium

# The default run (wall-clock) time is 1 minute
#SBATCH --time=10:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=4

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=20000

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# 90 seconds before training ends, to help create a checkpoint and requeue the job
#SBATCH --signal=SIGUSR1@90

module use /opt/insy/modulefiles

module load cuda/11.1 cudnn/11.1-8.0.5.39
module load miniconda/3.9

# Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
# For example: srun python my_program.py
# Use this simple command to check that your sbatch settings are working (verify the resources allocated in the usage statistics)

rnd_uuid=$(uuidgen)

source activate /home/nfs/oshirekar/unsupervised_ml/ai2
# srun python runner.py cactus --emb_data_dir="/home/nfs/oshirekar/unsupervised_ml/data/cactus_data" --n_ways=5 --n_shots=1 --use_precomputed_partitions=False

srun python ../runner.py protoclr_ae miniimagenet "/home/nfs/oshirekar/unsupervised_ml/data/" \
  --lr=1e-3 \
  --inner_lr=1e-3 \
  --eval-ways=5 \
  --eval_support_shots=5 \
  --batch_size=50 \
  --distance='euclidean' \
  --tau=1.0 \
  --logging='wandb' \
  --clustering_alg='None' \
  --ae=False \
  --use_umap=False \
  --profiler='simple' \
  --train_oracle_mode=False \
  --callbacks=False \
  --patience=200 \
  --no_aug_support=True \
  --ckpt_dir="/home/nfs/oshirekar/unsupervised_ml/ckpts" \
  --uuid=$rnd_uuid
