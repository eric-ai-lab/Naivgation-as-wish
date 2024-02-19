name=agent-env-fedavg-glr2_le5
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --features img_features/ResNet-152-imagenet.tsv
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 150000 --maxAction 35
      --if_fed True
      --comm_round 365
      --sample_fraction 0.2
      --global_lr 2
      --local_epoches 5

      --no_train True
      --backdoor_valid True
      --malicious_rate 1
      --backdoor_val_rate 0.1
      --seed 1
      --minus 1
      --n_parties 61
      --malicious_fraction 0.1
      --attack_type 3
      --backdoor_train_rate 0.3
      --scaled_factor 0.3
      --defense_method PBA"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python3 r2r_src/adv_train.py $flag --name $name