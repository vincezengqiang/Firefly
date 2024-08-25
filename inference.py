from transformers import AutoTokenizer, AutoConfig, AddedToken,AutoModelForCausalLM
import torch
from loguru import logger
import copy, json

import sys
import time, os

sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    # if tokenizer.__class__.__name__ == 'QWenTokenizer':
    #     tokenizer.pad_token_id = tokenizer.eod_id
    #     tokenizer.bos_token_id = tokenizer.eod_id
    #     tokenizer.eos_token_id = tokenizer.eod_id
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    if tokenizer.__class__.__name__ == 'QWen2TokenizerFast':
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = tokenizer.eos_token_id
        # tokenizer.eos_token_id = tokenizer.eos_token_id

    return tokenizer


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data


def write_jsonl(data, file_path, mode='w'):
    with open(file_path, mode, encoding='utf-8') as f:
        for json_data in data:
            json_line = json.dumps(json_data, ensure_ascii=False)
            f.write(json_line + '\n')


if __name__ == '__main__':
    # 使用合并后的模型进行推理
    # model_name_or_path = 'Qwen/Qwen-7B-Chat'
    # template_name = 'qwen'
    #  adapter_name_or_path = None
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f'torch.cuda.device_count():{torch.cuda.device_count()}')
    device = 'cuda'
    test_file = '/root/autodl-tmp/a00_Firefly/data/train_merge_cot_025.jsonl'
    des_file = '/root/autodl-tmp/a00_Firefly/data/res/train_merge_cot_025_res.jsonl'
    data = read_jsonl(test_file)

    model_name_or_path = '/root/autodl-tmp/a00_Firefly/backup/firefly-qwen-7b-sft-full/final'
    # model_name_or_path = '/root/autodl-tmp/Qwen2-7B-Instruct'
    adapter_name_or_path = None

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 300
    top_k = 10
    top_p = 0.9
    temperature = 1.0
    repetition_penalty = 1.0
    do_sample = False

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='auto',
        ).eval()
    model.generation_config.chat_format = 'chatml'
    model.generation_config.max_window_size = '8192'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    print(tokenizer.__class__.__name__)
    if tokenizer.__class__.__name__ == 'Qwen2TokenizerFast':
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = tokenizer.eos_token_id
    # if tokenizer.__call__.__name__=='QwenTokenizer':
    #     tokenizer.pad_token_id = tokenizer.eod_id
    #     tokenizer.bos_token_id = tokenizer.eod_id
    #     tokenizer.eos_token_id = tokenizer.eod_id

    right, all = 0, 0
    total_token, time_cost = 0, 0
    res = []
    for idx, data in enumerate(data):
        if 'label0' not in data:
            continue
        messages = data['conversations']
        system = messages[0]['value']
        user = messages[1]['value']
        label = messages[-1]['value']
        result = copy.deepcopy(data)

        with torch.no_grad():
            message = [
                {
                    'role': 'system',
                    'content': system
                },
                {
                    'role': 'user',
                    'content': user
                },
            ]
            text = tokenizer.apply_chat_template(message,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
            model_input = tokenizer([text], return_tensors='pt').to(device)
            bgn = time.time()
            generated_ids = model.generate(
                model_input.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=151645)
            ens = time.time()

            generated_id = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(
                    model_input.input_ids, generated_ids)
            ]

            output = tokenizer.batch_decode(generated_id,
                                            skip_special_tokens=True)[0]

            print(f'idx:{idx}')
            print(f'cost time:{round(ens - bgn, 2)}')
            # print(f'system:{system}')
            print(f'user:{user}')
            print(f'predict:{output}')
            print(f'label:{label}')

            if label == output:
                right += 1
            all += 1

            token_len = len(tokenizer.encode(output))
            total_token += token_len
            time_cost += round(ens - bgn, 2)
        result['output'] = output
        res.append(result)
        write_jsonl(res, des_file)

    print(f'date len:{len(data)}')
    print(f'inference time cost:{time_cost}')
    print(f'total token:{total_token}')
    print(f'tps:{round(all/time_cost, 2)}')
    print(f'des_file:{des_file}')
    print(f'right:{right}')
    print(f'all:{all}')
    print(f'accuracy:{round(right/all, 3)}')
