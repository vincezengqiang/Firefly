from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM
import torch
from loguru import logger
import copy, json

import sys
import time, os

sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data


path = '/root/autodl-tmp/a00_Firefly/data/res/train_merge_cot_025_res.jsonl'
data = read_jsonl(path)
right, all = 0, 0
for idx, d in enumerate(data):
    if idx < 2400:
        continue
    messages = d['conversations']
    label = messages[-1]['value']
    output = d['output']
    if label == output:
        right += 1
    all += 1

print(f'date len:{len(data)}')
print(f'right:{right}')
print(f'all:{all}')
print(f'accuracy:{round(right/all, 3)}')
