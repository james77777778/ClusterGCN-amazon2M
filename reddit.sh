#!/bin/bash
export METIS_DLL=~/.local/lib/libmetis.so
python main.py --dataset reddit --exp_num 1 --run_name reddit_sota --batch_size 20 --num_clusters_train 1500 --num_clusters_val 20 --num_clusters_test 1 --layers 4 --epochs 130 --lr 0.005 --hidden 512 --dropout 0.2 --test 1 --diag_lambda 0.0001