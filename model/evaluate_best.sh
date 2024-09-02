exp_low=$1
exp_high=$2
dataset=$3
eval_method=$4
LC_NUMERIC="en_US.UTF-8"
for exp in $(seq $exp_low $exp_high); do
    python evaluate_best.py "$dataset" "$exp" $eval_method
done