#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
mode=$1  # "baselines" or "finetunes"
dataset_path=$2  # Replace this with your file's name
checkpoint=$3
output_dir=$4
num_sample=$5
num_gpu=$6
datasetname=$7
sep_token=$8
use_header=$9

echo "sep_token:"$sep_token
echo "use_header:"$use_header
echo $datasetname
# #checkpoint="/path/to/Buggy-Aware-CodeLM/checkpoints/hugginface_checkpoints/codegen-350M-mono"
# checkpoint="Salesforce/codegen-350M-mono"

mkdir -p $output_dir
echo "Successfully created output directory $output_dir"
chmod +w $output_dir


if [ -f "$dataset_path" ]; then
    line_count=$(wc -l < "$dataset_path")
    echo "The number of lines in $dataset_path is: $line_count"

else
    echo "File $file does not exist."
fi

# Calculate the interval size
interval_size=$(($line_count/$num_gpu))

echo "Interval size: $interval_size"

start=0 
for ((i = 1; i <= $num_gpu; i++)); do
    interval_start=$start
    interval_end=$((interval_start + interval_size))

    # Adjust the last interval's end if needed
    if [ $i -eq $num_gpu ]; then
        interval_end=$((line_count - 1))
    fi
    
    if [ -z "$sep_token" ]; then
        python inference.py --gpu_id $((i-1)) --p_start $interval_start --p_end $interval_end \
                            --dataset_path $dataset_path --checkpoint $checkpoint \
                            --datasetname $datasetname \
                            --output_dir $output_dir --num_sample $num_sample >> $output_dir/run.log & 
        #echo "Interval $i: Start=$interval_start, End=$interval_end"
        #python inference_1.py --gpu_id 0 --p_start 0 --p_end 1 --dataset_path /path/to/Buggy-Aware-CodeLM/datasets/benchmarks/demo_test.jsonl --checkpoint /path/to/Buggy-Aware-CodeLM/checkpoints/hugginface_checkpoints/codegen-350M-mono --datasetname buggy_fixeval --output_dir /path/to/Buggy-Aware-CodeLM/results/finetune/codegen-350M-mono --num_sample 100
    else
        python inference.py --gpu_id $((i-1)) --p_start $interval_start --p_end $interval_end \
                            --dataset_path $dataset_path --checkpoint $checkpoint \
                            --datasetname $datasetname --sep_token $sep_token --use_header $use_header\
                            --output_dir $output_dir --num_sample $num_sample >> $output_dir/run.log & 
    fi
    start=$((interval_end + 1))  # Update start for the next interval       
done

wait

cd ${output_dir}/${datasetname}
cat *.jsonl > all.jsonl

echo "Combined JSONL files into ${output_dir}/${datasetname}/all.jsonl"