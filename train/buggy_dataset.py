# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json 
import random 
import os 
import json, gzip 
from typing import Iterable, Dict
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import read_problems

class BuggyDataset(Dataset):
    def __init__(self, train_files, tokenizer, train_size=None, clean=False, buggy=True, sep_token="[SEP]",end_token="<|endoftext|>", max_length=2048) -> None:
        if isinstance(train_files, str):
             self.total_data = list(read_problems(train_files))
        if isinstance(train_files, list):
            # have multiple different training datasets
            self.total_data = []
            for train_file in train_files:
                self.total_data.extend(list(read_problems(train_file)))
        print("The total number of samples is:", len(self.total_data))
        print("The specified training size is:", train_size)
        if train_size is not None:
            self.total_data = random.sample(self.total_data, train_size)


        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.label_ids = []
        self.buggy_ids = []

        # calculate the max length of the input ids 
        local_max_length = 0
        for item in self.total_data:
            buggy_dict = tokenizer(item["buggy_prompt"]+item["canonical_solution"])
            local_max_length = max(local_max_length, len(buggy_dict["input_ids"]))

        for item in self.total_data:
            if buggy:
                buggy_dict = tokenizer(item["buggy_prompt"] + sep_token, truncation=True, 
                                    max_length=min(local_max_length,max_length), return_tensors="pt")
                input_dict = tokenizer(item["buggy_prompt"] + sep_token + item["canonical_solution"] + end_token, padding="max_length", 
                                        truncation=True, max_length=min(local_max_length,max_length), return_tensors="pt")
                label = torch.clone(input_dict["input_ids"][0])
                label[:len(buggy_dict["input_ids"][0])] = tokenizer.pad_token_id
                self.buggy_ids.append(buggy_dict["input_ids"][0])
                self.input_ids.append(input_dict["input_ids"][0])
                self.label_ids.append(label)
                self.attn_masks.append(input_dict["attention_mask"])
                
            if clean:
                # clean data 
                buggy_dict = tokenizer(item["clean_prompt"] + sep_token, truncation=True, 
                                    max_length=min(local_max_length,max_length), return_tensors="pt")
                input_dict = tokenizer(item["clean_prompt"] + sep_token + item["canonical_solution"] + end_token, padding="max_length", 
                                        truncation=True, max_length=min(local_max_length,max_length), return_tensors="pt")
                label = torch.clone(input_dict["input_ids"][0])
                label[:len(buggy_dict["input_ids"][0])] = tokenizer.pad_token_id
                self.buggy_ids.append(buggy_dict["input_ids"][0])
                self.input_ids.append(input_dict["input_ids"][0])
                self.label_ids.append(label)
                self.attn_masks.append(input_dict["attention_mask"])
                assert len(label) == len(input_dict["input_ids"][0])
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_id": self.input_ids[idx], "label": self.label_ids[idx], "attention_mask": self.attn_masks[idx]}
  

if __name__ == '__main__':
    pass 