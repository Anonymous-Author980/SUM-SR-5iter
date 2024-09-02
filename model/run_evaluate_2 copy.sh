# -*- coding: utf-8 -*-
#freeze reconstructor
base_path="../../../gluster/exp-notgan-pretrain-3/reg0.7"
exp_low=$1
exp_high=$2
dataset=$3
LC_NUMERIC="en_US.UTF-8"
for exp in $(seq $exp_low $exp_high); do
    exp_path="$base_path/exp$exp"
    sh exportTensorLog2.sh $exp_path $exp_path $dataset
    python evaluate_2.py --video_type $dataset --exp $exp
done