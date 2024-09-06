# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Iterable
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
                
def remove_code(prompt, data_name):
    separator1 = '"""'
    separator2 = "'''"
    if separator1 in prompt:
        sep = separator1
    elif separator2 in prompt:
        sep = separator2
    else:
        print("Cannot find separator!!!")
        return None 

    if data_name.lower() == 'fixeval':
        lst_prompt = prompt.split(sep)
        s = lst_prompt[-1]
        prompt = sep.join(lst_prompt[:-1]) + sep
        lst = s.split('\n')
        n = len(lst)
        last_idx = 0
        for i in range(n-1, 0, -1):
            if 'input()' in lst[i]:
                last_idx = i
                break
        txt = prompt + '\n'.join(lst[:last_idx+1])
        return txt
    else:
        lst_prompt = prompt.split(sep)
        s = lst_prompt[-1]
        return sep.join(lst_prompt[:-1]) + sep
    
def remove_lines(code, indent=''):
    ls = code.rstrip().split('\n')
    lines = []
    for i in range(len(ls)):
        if ls[i].strip() != '':
            lines.append(ls[i][len(indent):])
    return '\n'.join(lines)

def truncate_str(gen_text):
    #truncate_before_pattern=["\n\n\n", r"\n\n^#", '"""', '\n\ndef', "^'''",  "<|endoftext|>", "</", "\nfrom", "\nimport", 'if __name__ ==', '\n\n']
    # truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]

    # they only do one!!!!!
    # import re
    # [m.start() for m in re.finditer('\ndef', x)]
    if "<|endoftext|>" in gen_text: # endoftext should be the strong indicator
        idx = gen_text.find("<|endoftext|>")
        return gen_text[:idx]
    
    # min_idx = 10000
    # for s in truncate_before_pattern:
    #     if s in gen_text:
    #         idx = gen_text.find(s)
    #         if idx > -1 and min_idx > idx:
    #             min_idx = idx
    # if min_idx < 10000:
    #     return gen_text[:min_idx]
    return gen_text

def estimate_batch_size(model_name, input_len):
    if "codegen" in model_name:
        if input_len <= 200:
            return 12
        elif input_len <= 257:
            return 8
        else:
            return 5
    elif "incoder-6B" in model_name:
        if input_len <= 200:
            return 15
        elif input_len <= 257:
            return 8
        else:
            return 5
    elif  "incoder-1B" in model_name:
        if input_len <= 200:
            return 30
        elif input_len <= 257:
            return 20
        else:
            return 10
    else:
        if input_len <= 200:
            return 40
        elif input_len <= 284:
            return 30
        else:
            return 20
