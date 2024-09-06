# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
deepspeed train_deepspeed.py \
       --checkpoint Salesforce/codegen-350M-mono \
       --deepspeed_checkpoint_dir /path/to/Buggy-Aware-CodeLM/checkpoints/deepspeed_checkpoints \
       --train_files /path/to/Buggy-Aware-CodeLM/datasets/train/train_40k.jsonl