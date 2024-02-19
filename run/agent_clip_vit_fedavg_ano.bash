name=agent-fedavg-glr4_le3
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 150000 --maxAction 35
      --if_fed True
      --comm_round 365
      --sample_fraction 0.2
      --global_lr 12
      --local_epoches 5
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python3 redo/train.py $flag --name $name
