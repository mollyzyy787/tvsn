#!/bin/bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 th train_cdoafn.lua --category car --modelString CDOAFN_SYM --imgscale 256 --batchSize 25 --lr 0.00005 --saveFreq 1 --maxEpoch 1 --gpu 0 --nThreads 8 --resume 0
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 th train_cdoafn.lua --category car --modelString CDOAFN_SYM --imgscale 256 --batchSize 25 --lr 0.00001 --saveFreq 1 --maxEpoch 2 --gpu 0 --nThreads 8 --resume 1
