srun python ../../runner.py protoclr_ae miniimagenet "./data/" \
  --lr=1e-3 \
  --inner_lr=1e-3 \
  --batch_size=200 \
  --num_workers=6 \
  --eval-ways=5 \
  --eval_support_shots=1 \
  --distance='euclidean' \
  --logging='wandb' \
  --clustering_alg="hdbscan" \
  --km_clusters=5 \
  --cl_reduction="mean" \
  --ae=False \
  --profiler='simple' \
  --train_oracle_mode=False \
  --callbacks=False \
  --patience=200 \
  --estop_ckpt_on_val_acc=True \
  --no_aug_support=True \
  --ckpt_dir="./ckpts" \
  --use_umap=False \
  --rerank_kjrd=True \
  --rrk1=20 \
  --rrk2=6 \
  --rrlambda=0