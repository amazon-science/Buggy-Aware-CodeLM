# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
import torch 
from tqdm import tqdm
from typing import Iterable, Dict
import gzip, os 
from utils import read_problems, write_jsonl, remove_code, remove_lines, truncate_str, estimate_batch_size

parser = argparse.ArgumentParser()
parser.add_argument("--p_start", default=0, type=int)
parser.add_argument("--p_end",  type=int)
parser.add_argument("--mode",  type=str, default="buggy", choices=['clean', 'buggy'])
parser.add_argument("--gpu_id",  default=0, type=int)
parser.add_argument("--num_samples", default=100, type=int)
parser.add_argument("--datasetname", default="buggy_fixeval", type=str)
parser.add_argument("--sep_token", default="", type=str)
parser.add_argument("-t","--temperature", default=0.6, type=float)
parser.add_argument("--use_header", type=int, help="use header or not")
parser.add_argument("--batch_size", type=int, help="batch size for generation")
parser.add_argument("--iteration_k", type=int, default=1, help="the number of iterations")
parser.add_argument("--use_partial", type=int, default=0, help="use the full generated content or only part of the generated content")
parser.add_argument("--checkpoint", default="Salesforce/codegen-350M-mono", type=str)
parser.add_argument("--dataset_path", default="datasets/benchmarks/demo_test.jsonl",  type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()


def load_model(checkpoint, device):
    # model
    if '6B' in checkpoint:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model = model.half().to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
         print("Tokenizen has no pad token, adding [PAD] to the vocabulary")
         tokenizer.add_special_tokens({'pad_token': '[PAD]','sep_token': "[SEP]"})
    model.eval()
    return model, tokenizer

###================== SETUP  ===================######
if args.gpu_id > 0:
    device = torch.device(f"cuda:{args.gpu_id}")
else: 
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

max_lens = {'buggy_humaneval': {'gen': 300, 'ref': 300, 'full': 650}, 'buggy_fixeval': {'gen': 128, 'ref': 256, 'full': 600}}
max_new_tokens =  max_lens[args.datasetname]['gen']
max_full_length = max_lens[args.datasetname]['full']


###================== DATA & SETTING ===================######
problems_sets = read_problems(args.dataset_path)
if args.p_start is not None and args.p_end is not None:
    problems = [problems_sets[i] for i in range(args.p_start, args.p_end+1)]
    completion_file = f'{args.output_dir}/{args.datasetname}/{args.p_start}-{args.p_end}_{os.path.basename(args.checkpoint)}.jsonl'
else:
    problems = list(problems_sets)
    completion_file = f'{args.output_dir}/{args.datasetname}/all_{os.path.basename(args.checkpoint)}.jsonl'

os.makedirs(os.path.dirname(completion_file), exist_ok=True)
n_problems = len(problems)

header_lst, task_id_lst, instance_id_lst, prompt_lst = [], [], [], []
for i, problem in enumerate(problems):
    task_id_lst.append(problem['task_id'])
    header_lst.append(problem["prompt"])
    instance_id_lst.append(problem['instance_id']) #instance_id
    prompt_lst.append(remove_lines(problem[f'{args.mode}_prompt'], ''))

if args.method == "removal":
    prompt_lst = [remove_code(prompt, args.dataset) for prompt in prompt_lst]
    
###================== COMPLETION ===================######
model, tokenizer = load_model(args.checkpoint, device)
samples = []
fail_ids = []

task_id_set = set()
try:
    for i in range(n_problems):
        if task_id_lst[i] in task_id_set:
            continue
        task_id_set.add(task_id_lst[i])
        print("Problem:", i)
        new_samples = []
        header, instance_id, task_id, prompt = header_lst[i], instance_id_lst[i], task_id_lst[i], prompt_lst[i]
        print("task_id: ", task_id, "instance_id: ", instance_id)
        try:
            # completions = generate(model, tokenizer, prompt, args.temperature, args.batch_size, args.num_samples, device)
            #### generate ####
            completions = []
            tokens = tokenizer([prompt+args.sep_token], return_tensors='pt') # no truncation
            input_ids=tokens['input_ids'].to(device)
            input_len = input_ids[0].flatten().size(0)
            num_generated_tokens = max_new_tokens
            batch_size = estimate_batch_size(args.checkpoint, input_len) if args.batch_size is None else args.batch_size
            total_steps = (args.num_samples//args.iteration_k - 1)// batch_size + 1
            print("total_steps:", total_steps, "batch_size:", batch_size)
            if total_steps <= 0:
                total_steps = 1
            
            next_out_texts = []
            for j in tqdm(range(total_steps), desc=f"Problem {i}"):
                bz = min(batch_size, args.num_samples//args.iteration_k - batch_size * j)
                set_input_ids = input_ids.expand(bz, -1)
                with torch.no_grad():
                    if bz > 1:
                        if 'codegen' in args.checkpoint:
                            outs = model.generate(input_ids=set_input_ids, max_new_tokens= max_full_length if args.use_header else num_generated_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=args.temperature)
                        else:
                            outs = model.generate(input_ids=set_input_ids, do_sample=True, top_p=0.95, temperature=0.2, max_new_tokens=max_full_length if args.use_header else num_generated_tokens)
                    else:
                        if 'codegen' in args.checkpoint:
                            outs = model.generate(input_ids=set_input_ids, max_new_tokens=max_full_length if args.use_header else num_generated_tokens)
                        else:
                            outs = model.generate(input_ids=set_input_ids, max_new_tokens=max_full_length if args.use_header else num_generated_tokens)

                    outs = tokenizer.batch_decode(outs[:, input_len:], clean_up_tokenization_spaces=False, skip_special_tokens=True)
                    
                    out_texts = [truncate_str(o) for o in outs]
                    next_out_texts.extend(out_texts)
                    completions += out_texts
            
            # this is for inference 2 when the iteration_ is not 1
            for k in range(args.iteration_k-1):
                new_prompt = [header + next_out_text for next_out_text in next_out_texts]
                tokens = tokenizer(new_prompt, padding=True,  return_tensors='pt')
                if args.use_partial:
                    input_ids=tokens['input_ids'][:,:input_len].to(device)
                else:
                    input_ids=tokens['input_ids'].to(device)
                new_outputs = []
                for tmp_i in range((input_ids.size(0)-1) // (batch_size+1)):
                    tmp_ids = input_ids[:batch_size + batch_size * tmp_i]
                    with torch.no_grad():
                        # using the gready search 
                        if 'codegen' in args.checkpoint:
                            outs = model.generate(input_ids=tmp_ids, max_new_tokens=max_full_length if args.use_header else num_generated_tokens, pad_token_id=tokenizer.eos_token_id, num_beams=3)
                        else:
                            outs = model.generate(input_ids=tmp_ids, num_beams=3, max_new_tokens=max_full_length if args.use_header else num_generated_tokens)
                            
                        outs = tokenizer.batch_decode(outs, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                        cut_outs = [out[len(new_prompt[i]):] if out.strip() != new_prompt[i].strip() else new_prompt[i][len(header):] for i, out in enumerate(outs)]
                        out_texts = [truncate_str(o) for o in cut_outs]
                        next_out_texts.extend(out_texts)
                        completions += out_texts

            for c in completions:
                if args.use_header:
                    print("use_header:", args.use_header)
                    program = header + '\n' + c
                else:
                    program = prompt + '\n' + c
                new_samples.append(dict(instance_id=instance_id, task_id=task_id, completion=program))
            
        except RuntimeError as e:
            fail_ids.append(instance_id)
            print('Stop because of CUDA mem ')     
            del model
            torch.cuda.empty_cache() 
            print('Restart model')   
            model, tokenizer = load_model(args.checkpoint, device)
            # break
        ## append after each run from the 2nd run
        samples += new_samples
        write_jsonl(completion_file, new_samples,append=True)

except Exception as e: # work on python 3.x
    print('Failed '+ str(e))

print('Failed IDs: ' + ' '.join(fail_ids))