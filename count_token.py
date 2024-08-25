from transformers import AutoTokenizer
from component.dataset import (
    UnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset
)
from component.template import template_dict

src_path='/root/autodl-tmp/a00_Firefly/data/train_merge_cot_025.jsonl'
tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen2-7B-Instruct',trust_remote_code=True)
template = template_dict['qwen']

train_datasets=UnifiedSFTDataset(src_path, tokenizer, 8192, template)

max_len=0
total=0
for i in range(train_datasets.__len__()):
    inputs=train_datasets.__getitem__(i)
    len_=len(inputs['input_ids'])
    max_len=max(len_,max_len)
    total+=len_
print(f'max_len:{max_len}')
print(f'total:{total}')