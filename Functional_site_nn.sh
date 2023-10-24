#!/bin/bash
#SBATCH --job-name=prot_func_site
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:05:00
hostname
echo $CUDA_VISIBLE_DEVICES
python Functional_site_nn.py \
    --data_input /home/qcx679/hantang/github/_2022_functional-sites-cagiada/output_full.csv \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 100 \
    --target_type regression > Orig_NN.log;
echo "Done"
