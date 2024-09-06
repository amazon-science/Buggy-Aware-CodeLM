# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import Dict, Iterable
from collections import defaultdict
import json 
import gzip 
import os 

def read_problems(evalset_file) -> Dict[str, Dict]:
    return [instance for instance in stream_jsonl(evalset_file)]


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
                    
def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


parser = argparse.ArgumentParser()
parser.add_argument("--original_file", type=str, help="the path to the input file for the inference")
parser.add_argument("--completion_file",  type=str, help="the path to the inference result file")
parser.add_argument("--prefix",  type=str, help="a customized prefix for the instance id")
parser.add_argument("--save_path",  type=str, help = "the path for saving the newly-constructed dataset")
args = parser.parse_args()

if not os.path.exists(args.original_file):
    raise Exception("{args.original_file} do not exist!")

if not os.path.exists(args.completion_file):
    raise Exception("{args.completion_file} do not exist!")

    
ori_data = list(read_problems(args.original_file))
completion_data = list(read_problems(args.completion_file))

ori_dict = defaultdict(list)
for item in ori_data:
    ori_dict[item["instance_id"]] = [item["buggy_prompt"], item["canonical_solution"]]

results = []
instance_id = 0
for item in completion_data:
    completion = item["completion"].split("\n")
    buggy_prompt_lines = len(ori_dict[item["instance_id"]][0].split("\n"))
    new_buggy_prompt = "\n".join(completion[:buggy_prompt_lines])
    results.append(
        {
            "instance_id": f"{args.prefix}/{instance_id}",
            "buggy_prompt": new_buggy_prompt,
            "canonical_solution": ori_dict[item["instance_id"]][1]
        }
    )
    instance_id += 1

write_jsonl(args.save_path, results)