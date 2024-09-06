# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os, json
from typing import Iterable, Dict
import gzip
from collections import defaultdict
from transformation_classes import *
import ast
import random
import json 


data_name_path = json.load(open("meta.json"))["datapath"]
TRANSFORMATION_CLASSES = json.load(open("meta.json"))["method"]
save_path_dir = json.load(open("meta.json"))["save_path_dir"]
def calculate_indent(code):
    pre_indent_buffer = "" # get the indent for the line ahead this split line
    for ch in code[code.rfind('\n') + 1:]:
        if ch in [" ", "\t"]:
            pre_indent_buffer += ch
        else:
            break
    return pre_indent_buffer


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def read_problems(evalset_file) -> Dict[str, Dict]:
    return [instance for instance in read_jsonl(evalset_file)]


def read_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def generate_buggy():
    from tqdm import tqdm 
    fail_id = 0 
    for datasetname, datapath in data_name_path.items():
        datasets = read_problems(datapath)
        for method in TRANSFORMATION_CLASSES:
            results = []
            for item in tqdm(datasets, desc=f"{datasetname}: {method}"):
                buggy_item = item.copy()
                complete_code = item["prompt"] + item["canonical_solution"]
                # remove some redundant lines
                try:
                   ast_tree = ast.parse(complete_code)
                except:
                    # unable to pass the ast parser
                    fail_id += 1
                    continue 
                complete_code = ast.unparse(ast_tree)
                total_line_number =  len(complete_code.split("\n"))
                exclude_lineno = []
                instance_id = 1
                maximum_trials = 0
                while True:
                    try:
                       ast_tree = ast.parse(complete_code)
                    except:
                        break
                    modifier = eval(method)(exclude_lineno=exclude_lineno)
                    try:
                        result = modifier.visit(ast_tree)
                    except:
                        continue

                    if "Visitor" in method:
                        result = ast_tree
                
                    if modifier.linenumber is not None:
                        exclude_lineno.append(modifier.linenumber)
                        try:
                            buggy_complete = ast.unparse(result)
                        except:
                            print(
                                "The buggy code {} is not parsable!!".format(
                                    item["task_id"]
                                )
                            )
                            continue
                        new_item = dict()
                        new_item["task_id"] = item["task_id"]
                        new_item["line_number"] = modifier.linenumber
                        new_item["buggy_full_code"] = buggy_complete
                       
                        
                        # make the buggy appear in different positions
                        offset = random.choice([i for i in range(max(1, total_line_number-modifier.linenumber))])
                        truncated_index = modifier.linenumber + offset 
                        buggy_prompt = "\n".join(buggy_complete.split("\n")[: truncated_index])
                        buggy_item["buggy_prompt"] = buggy_prompt
                        buggy_item["instance_id"] = method + "_" + datasetname + "_" + str(instance_id)
                        buggy_item["method"] = method
                        results.append(buggy_item.copy())
                        instance_id += 1
                        #complete_code = buggy_complete # continue to inject bugs in a buggy code

                        maximum_trials += 1
                        if maximum_trials > 2:
                            break
                    else:
                        break

            os.makedirs(os.path.join(save_path_dir, datasetname), exist_ok=True)
            write_jsonl(
                os.path.join(save_path_dir, datasetname, f"{method}.jsonl"),
                results,
            )


def generate_clean():
    for datasetname, datapath in data_name_path.items():
        datasets = read_problems(datapath)

        results = []
        for item in datasets:
            buggy_item = item.copy()
            canonical_solution = item["canonical_solution"]
            lines = canonical_solution.split("\n")
            num_lines = canonical_solution.count("\n")
            candidates_partials = set()

            truncated_indexes = [int(num_lines * i) for i in [0.2, 0.4, 0.6, 0.8]]

            for i in truncated_indexes:
                partial_code = "\n".join(lines[: i])
                candidates_partials.add(item["prompt"] + partial_code)

            instance_id = 1
            for buggy_prompt in candidates_partials:
                buggy_item["buggy_prompt"] = buggy_prompt
                buggy_item["instance_id"] = "clean_" + datasetname + "_" +  str(instance_id)
                results.append(buggy_item.copy())
                instance_id += 1

        write_jsonl(
            os.path.join(save_path_dir, f"{datasetname}_clean.jsonl"),
            results,
        )
        
if __name__ == "__main__":
    generate_buggy()
    generate_clean()
   

    

