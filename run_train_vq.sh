#!/bin/bash

# grvq_q8_g2_cs512_cd1024
nohup python train_vq.py --name grvq_q8_g2_cs512_cd1024 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 8 --vq_group 2 --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --train_env "local" --vq_arch_option "group_residual_vq" --nb_code 512 --code_dim 1024 > nohup_grvq_q8_g2_cs512_cd1024_output.txt 2>&1

# grvq_q8_g4_cs512_cd1024
nohup python train_vq.py --name grvq_q8_g4_cs512_cd1024 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 8 --vq_group 4 --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --train_env "local" --vq_arch_option "group_residual_vq" --nb_code 512 --code_dim 1024 > nohup_grvq_q8_g4_cs512_cd1024_output.txt 2>&1

# grvq_q8_g8_cs512_cd1024
nohup python train_vq.py --name grvq_q8_g8_cs512_cd1024 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 8 --vq_group 8 --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --train_env "local" --vq_arch_option "group_residual_vq" --nb_code 512 --code_dim 1024 > nohup_grvq_q8_g8_cs512_cd1024_output.txt 2>&1

# grvq_q4_g2_cs512_cd1024
nohup python train_vq.py --name grvq_q4_g2_cs512_cd1024 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 4 --vq_group 2 --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --train_env "local" --vq_arch_option "group_residual_vq" --nb_code 512 --code_dim 1024 > nohup_grvq_q4_g2_cs512_cd1024_output.txt 2>&1

# grvq_q4_g4_cs512_cd1024
nohup python train_vq.py --name grvq_q4_g4_cs512_cd1024 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 4 --vq_group 4 --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --train_env "local" --vq_arch_option "group_residual_vq" --nb_code 512 --code_dim 1024 > nohup_grvq_q4_g4_cs512_cd1024_output.txt 2>&1

# grvq_q4_g8_cs512_cd1024
nohup python train_vq.py --name grvq_q4_g8_cs512_cd1024 --gpu_id 0 --dataset_name kit --batch_size 256 --num_quantizers 4 --vq_group 8 --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --train_env "local" --vq_arch_option "group_residual_vq" --nb_code 512 --code_dim 1024 > nohup_grvq_q4_g8_cs512_cd1024_output.txt 2>&1
