# -*- coding: utf-8 -*-
#freeze reconstructor
exp_low=$1
exp_high=$2
dataset=$3
n_epochs=$4
LC_NUMERIC="en_US.UTF-8"
for exp in $(seq $exp_low $exp_high); do
    python evaluate_2.py --video_type "$dataset" --exp $exp --n_epochs $n_epochs
done