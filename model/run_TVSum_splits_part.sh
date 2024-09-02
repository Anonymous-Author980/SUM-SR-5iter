# -*- coding: utf-8 -*-
#freeze reconstructor
basepath='../../../gluster/exp-notgan-pretrain-5iter/reg0.7'
exp_low=$1
exp_high=$2
summary_rate=$3
n_epochs=$4
iter_low=$5
iter_high=$6
LC_NUMERIC="en_US.UTF-8"
for exp in $(seq $exp_low $exp_high); do
    for i in $(seq 0 4); do
        for j in $(seq $iter_low $iter_high); do
            python main.py --split_index $i --video_type 'TVSum' --full_mode 'part' --exp $exp --summary_rate $summary_rate --iter $j --n_epochs $n_epochs --recon_n_epochs $n_epochs
            path1="$basepath/exp$exp/iter$j/TVSum/logs/split$i"
            python exportTensorFlowLog.py "$path1" "$path1"
            path2="$basepath/exp$exp"
            python evaluate_unsupervisedly.py "$path2" "TVSum" $j $i "avg" $n_epochs
        done
    done
done
