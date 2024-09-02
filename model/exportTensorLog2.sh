base_path=$1
base2_path=$2
dataset=$3
LC_NUMERIC="en_US.UTF-8"

exp_path="$base_path"; echo "$exp_path"
for i in 0 1 2 3 4; do
    path="$exp_path/$dataset/logs/split$i"
    path2="$base2_path/$dataset/logs/split$i"
    python exportTensorFlowLog.py "$path" "$path2"
done
