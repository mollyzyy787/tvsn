#!/bin/bash
OMP_NUM_THREADS=2 th train_ctvsn_maskcor.lua --category car --modelString CTVSN_maskcor --lambda1 100 --lambda2 0.001 --background 0 --iterG 2 --tv_weight 0.0001 --pixel_weight 0 --lr 0.0001 --beta1 0.5 --imgscale 256 --batchSize 15 --saveFreq 20 --maxEpoch 100 --gpu 0 --nThreads 0 --resume 0
OMP_NUM_THREADS=2 th train_ctvsn_maskcor.lua --category car --modelString CTVSN_maskcor --lambda1 100 --lambda2 0.001 --background 0 --iterG 2 --tv_weight 0.0001 --pixel_weight 0 --lr 0.00001 --beta1 0.5 --imgscale 256 --batchSize 15 --saveFreq 20 --maxEpoch 200 --gpu 0 --nThreads 0 --resume 1
OMP_NUM_THREADS=2 th train_ctvsn_maskcor.lua --category car --modelString CTVSN_maskcor --lambda1 100 --lambda2 0.001 --background 0 --iterG 2 --tv_weight 0.0001 --pixel_weight 10 --lr 0.00001 --beta1 0.5 --imgscale 256 --batchSize 15 --saveFreq 10 --maxEpoch 220 --gpu 0 --nThreads 0 --resume 1
