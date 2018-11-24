#!/bin/bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 th train_doafn2.lua --category car --modelString DOAFN2_SYM --imgscale 256 --batchSize 1 --lr 0.00005 --saveFreq 1 --maxEpoch 2 --gpu -1 --nThreads 0 --resume 0
#OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 th train_doafn.lua --category car --modelString DOAFN_SYM --imgscale 256 --batchSize 25 --lr 0.00001 --saveFreq 20 --maxEpoch 200 --gpu 0 --nThreads 0 --resume 1
