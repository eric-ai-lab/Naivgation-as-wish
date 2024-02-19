name=agent_rxr_en_clip_vit_fedavg_new_glr2
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 400000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 2
      --comm_round 910
      --attack_type 3
      --backdoor_train_rate 0.3
      --backdoor_val_rate 0.1
      --scaled_factor 0.1
      --malicious_fraction 0.1
      --defense_method PBA
      --local_epoches 5
      --do_mask True
      --n_parties 60
      --minus 1
      --no_train True
      --malicious_rate 1
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 rxr_src/adv_train.py $flag --name $name