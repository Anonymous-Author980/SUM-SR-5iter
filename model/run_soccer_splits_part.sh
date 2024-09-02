# -*- coding: utf-8 -*-
#freeze reconstructor
basepath='../../../old/exp-notgan-pretrain-5iter-others'
exp_low=$1
exp_high=$2
summary_rate=$3
n_epochs=$4
LC_NUMERIC="en_US.UTF-8"
for exp in $(seq $exp_low $exp_high); do
    for i in $(seq 0 4); do
        for j in $(seq 0 4); do
            python main.py --split_index $i --video_type 'soccer' --full_mode 'part' --exp $exp --summary_rate $summary_rate --iter $j --n_epochs $n_epochs --recon_n_epochs $n_epochs
            path1="$basepath/exp$exp/iter$j/soccer/logs/split$i"
            python exportTensorFlowLog.py "$path1" "$path1"
            path2="$basepath/exp$exp"
            python evaluate_unsupervisedly.py "$path2" "soccer" $j $i "max" $n_epochs
        done
        python pick_best_iter.py "$path2" "soccer" $i
    done
done
