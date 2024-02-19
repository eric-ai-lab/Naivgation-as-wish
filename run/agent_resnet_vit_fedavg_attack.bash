name=agent_clip_vit_fedavg_new_glr2_epc1_attack
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 150000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --attack_type 3
      --backdoor_train_rate 0.3
      --scaled_factor 0.1
      --defense_method PBA
      --global_lr 2
      --n_parties 61
      --no_train True
      --malicious_rate 1
      --backdoor_val_rate 0.1
      --do_mask True
      --seed 1
      --malicious_fraction 0.1
      --minus 1
      --local_epoches 5"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 r2r_src/adv_train.py $flag --name $name

# --load snap/agent_clip_vit_fedavg_new/state_dict/latest_dict