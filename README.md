# Fine-tuning Language Models for Joint Rewriting and Completion of Code with Potential Bugs

This is the experiment code for our ACL 2024 (Findings) paper "Fine-tuning Language Models for Joint Rewriting and Completion of Code with Potential Bugs".

Our constructed testing datasets are located in [Benchmarks](datasets/benchmarks).


## Buggy Dataset Generation

We provide a configuration file [meta.json](generator/meta.json), where you can specify the original datasource path and the bug injection method you want to use:

```json
{
    "datapath": {
        "codecontests": "/path/to/Buggy-Aware-CodeLM/datasets/raw/codecontests.jsonl"
    },
    "method" : [
            "OperatorChangeNodeVisitor",
            "NumericValueChangeNodeVisitor",
            "VariableRenamingNodeVisitor",
            "KeywordRemovalTransformer",
            "ConditionRemovalNodeVisitor",
            "BranchRemovalNodeVisitor",
            "WhileToIfTransformer"
        ],
    "savepath": "/path/to/Buggy-Aware-CodeLM/datasets/buggy"
 }

```

Then, you can run the following command to generate the buggy datasets, which will be save in the specified savepath in the **meta.json** file. 

``` python 
python generate.py
```
To prepare the dataset for the second-phase of training, begin by obtaining a fine-tuned model using the previously constructed training dataset. Once you have the fine-tuned model, you can commence the inference process with this model. Assuming the inference results are saved in a file named **result_1.jsonl**, you can generate the new training dataset by executing the following command:

```python
python truncate.py --original_file /path/to/Buggy-Aware-CodeLM/datasets/benchmarks/s_humaneval.jsonl --completion_file infill_line_completion_100_0-1894.jsonl --prefix iteration1 --save_path D_1.jsonl
```

Afterwards, you can proceed with further inference using both the fine-tuned model and the newly constructed dataset stored at save_path **D_1.jsonl**. This will yield additional training data.

......

Finally, you can combine all newly-constructed datasets and continue with the process of fine-tuning the model.


## Fine-tuning with DeepSpeed

```
deepspeed train_deepspeed.py \
       --checkpoint base_model_path or the huggingface model name \
       --deepspeed_checkpoint_dir deepspeed_save_path_dir \
       --train_files training_datafile_path
```
where **--train_files** argument supports passing multiple training files path. More argument choices could be found in [**train_deepspeed.py**](train/train_deepspeed.py) and [**ds_config.json**](train/ds_config.json).


## Inference with mutliple GPUS
We provide a [bash script](inference/inference.py) to make full use of the avaliable GPUs to speed up the inference. The main idea is to equally split the inference data to all available GPUs and then combine the inference results together. In particular, we provide the following arguments:

1. baseline or finetune 
2. the path for the dataset needed to be evaluated
3. checkpoint path
4. the directory for storing the results
5. number of samples we want to generate for each instance
6. number of gpus we want to use
7. datasetname, e.g. buggy_humaneval, buggy_fixeval
8. sep token
9. whether using the header

you can run the following command according to the above positional arguments:
```bash
bash inference.sh \
     finetune \
     /path/to/Buggy-Aware-CodeLM/datasets/benchmarks/fixeval_large_instances.jsonl \
     /path/to/Buggy-Aware-CodeLM/checkpoints/hugginface_checkpoints/codegen-350M-mono \
     /path/to/Buggy-Aware-CodeLM/results/finetune/codegen-350M-mono \
     1 \
     4 \
     buggy_fixeval \
     [SEP] \
     1 \
```

## License
The code in this package is subject to [Apache-2.0 License](LICENSE). 

The testing datasets in this repo are subject to different licenses:

- buggy-HumanEval files (`datasets/benchmarks/b_humaneval*.jsonl`) are released under the [MIT License](datasets/benchmarks/MIT.md).

- buggy-MBPP files (`datasets/benchmarks/b_mbpp*.jsonl`) are released under the [CC-BY-4.0 license](datasets/benchmarks/CC-BY-NC-4.0.md).


## Citation
You are more than welcome to cite our paper:
```
@inproceedings{wang2024fine,
  title={Fine-tuning Language Models for Joint Rewriting and Completion of Code with Potential Bugs},
  author={Wang, Dingmin and Zhao, Jinman and Pei, Hengzhi and Tan, Samson and Zha, Sheng},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={15854--15868},
  year={2024}
}
```