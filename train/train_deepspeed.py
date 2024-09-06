# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal example of training the 16B checkpoint on GPU with CPU offloading using deepspeed.

'''
apt install python3.8 python3.8-venv python3.8-dev

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0

deepspeed --num_gpus=1 train_deepspeed.py
'''

########################################################################################################
import deepspeed
from tqdm import tqdm 
import os
import argparse
import random
import torch.nn.functional as F
from time import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
from buggy_dataset import BuggyDataset
from datetime import datetime
########################################################################################################
## args


def calculate_loss(logits, labels, ignore_index):
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=ignore_index)
    return loss 


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help='local_rank')
    parser.add_argument("--seed", default=42, type=int, help='seed value')
    parser.add_argument("--checkpoint", required=True, type=str, help="a huggineface transformer modelname")
    parser.add_argument("--deepspeed_config", default="ds_config.json", type=str, help="the deepspeed config file path")
    parser.add_argument("--train_files", nargs='+',  required=True, type=str, help="the training dataset path")
    parser.add_argument("--deepspeed_checkpoint_dir", type=str, required=True, help="the deepspeed checkpoint base path")
    parser.add_argument("--epochs", type=int, default=5, help="the number of epochs")
    parser.add_argument("--use_wandb", type=int, default=0, help="whether use the wandb")
    

    return parser.parse_args()

def train(args):
    if args.use_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="buggy-training",
            group = os.path.basename(args.checkpoint)+"-"+datetime.now().strftime("%y-%m-%d"),
            # Track hyperparameters and run metadata
            config=vars(args))
    #######################
    ## preamble
    set_seed(args.seed)
    #######################
    ## model

    print('initializing model')

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
         tokenizer.add_special_tokens({'pad_token': '[PAD]','sep_token': "[SEP]"})
    model.resize_token_embeddings(len(tokenizer))
    train_datasets = BuggyDataset(args.train_files, tokenizer)

    model.train()
    # TODO(enijkamp): we need to set this flag twice?
    model.gradient_checkpointing_enable()

    #######################
    ## deepspeed

    print('initializing deepspeed')
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(config=args.deepspeed_config,
                                                                         model=model, model_parameters=model_parameters, 
                                                                         training_data=train_datasets)
    #######################
    ## train

    # print('starting training')
    # if os.path.exists(args.deepspeed_checkpoint_dir + "/" + os.path.basename(args.checkpoint)):
    #     import shutil
    #     shutil.rmtree(args.deepspeed_checkpoint_dir + "/" + os.path.basename(args.checkpoint))
    #     print("Delete the checkpoints:", args.deepspeed_checkpoint_dir + "/" + os.path.basename(args.checkpoint))
    #input_ids = torch.randint(low=0, high=10, size=[args.deepspeed_config['train_micro_batch_size_per_gpu'], 1024], dtype=torch.int64).cuda()
    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            # loss = model_engine(input_ids=input_ids, labels=input_ids).loss
            input_ids = batch["input_id"].cuda()
            labels = batch["label"].cuda()
            logits = model_engine(input_ids=input_ids).logits

            loss = calculate_loss(logits,labels, tokenizer.pad_token_id)
            if args.use_wandb:
                wandb.log({f"loss": loss.item()}, step=step + epoch * len(train_dataloader))

            model_engine.backward(loss)
            model_engine.step()

        save_dir = args.deepspeed_checkpoint_dir + "/" + os.path.basename(args.checkpoint) + f'/epoch_{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        model_engine.save_checkpoint(save_dir, "global_step_"+str(epoch))

        model_engine.train()

########################################################################################################
## preamble

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


########################################################################################################
## main

def main():
    # args
    args = create_args()
    # train
    train(args=args)

if __name__ == '__main__':
    main()
 