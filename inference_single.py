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
    model_name_or_path = '/root/autodl-tmp/firefly-qwen-7b-sft-full/final'
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


    system="##角色设定##\n- 身份：一个资深的安全生产专家\n- 能力：具备深入的安全生产经验、熟悉各种安全生产规则、能够指出各种安全生产问题\n- 目标：找到输入的安全生产'问题和隐患描述'对应的问题种类\n\n##任务##\n- 根据\"问题和隐患描述\"并参考”问题分类标准\"输出对应的问题分类标签\n\n##任务要求##\n1.根据输入的“问题和隐患描述“找到“问题分类标准“中最合适的标签，并输出。\n2.直接输出对应的标签，不要输出多余内容。\n\n##问题分类标签##\n设施问题、管理问题、人员能力问题、环境保护问题、个人防护用品问题、施工工程问题、作业场所问题\n\n##问题分类标准##\n1、设施问题：包括安防设备设施、电气设备设施、计量设备设施、管道及辅助设施、生产设备设施、生产辅助设施、特种设备、施工机具问题。具体问题包括：\n    1.1：固定式、便携式气体监测仪，入侵报警，视频监控，甲烷监控，安全阀，空呼，风机，接地，火焰监测、熄火保护相关问题。包括：系统不完善，安装不规范，本质故障，气体检测仪未检定，空呼未检定，安全阀未检定，参数设置不合理，工艺纪律执行不到位，接地不规范，巡检维护不到位.\n    1.2：配电室、配电间、配电箱、接线箱、配电柜、控制柜、控制器、机柜、电缆桥架、变压器、照明（不包括应急照明，应急照明属于消防管理）、电缆、接线盒、UPS（蓄电池）、插头、防爆挠线管相关问题。包括：本质故障，不满足防火防爆要求，工艺纪律执行不到位，用电行为不规范，防护设施缺失，敷（架）设不规范，防水条件不满足，选型不当，巡检维护不到位。\n    1.3：压力表、温度计、压力变送器、温度传感器、流量计、孔板阀、调压阀、露点检测仪、液位计、检测管（硫化氢）、游标卡尺、刀口尺、千分尺相关问题。包括：本质故障，泄漏，安装不规范，计量器具未检定，使用不规范，选型不当，未配备或使用。\n    1.4：站内管道、采气管道、输气管道、站外气田水管线、阴保、堡坎、标志桩、警示牌、警示带、点位测试桩、管道巡检、第三方施工相关问题。包括：本质故障、泄漏、标识错误、标识缺失、巡检维护不到位、风险未消除、系统不完善、埋深不规范、停用管理不规范\n    1.5：与生产直接相关的，天然气及生活用气、气田水经过的设备设施。例如井口装置、井安系统、水套炉、增压装置、脱水装置、气田水转运回注装置、站内管道、电磁阀、电动阀、气液联动球阀、火炬、爆破片、阻火器相关问题。包括：系统不完善、本质故障、安装不规范、附件缺失或失效。\n    1.6：与生产有关，但是天然气、气田水不经过的设备，例如：井口方井、增压厂房、值班室、泵房、仪表风系统、转水系统、发电机、药剂加注系统相关问题。包括：本质故障、风险未消除、附件缺失或失效、功能不完善、停用管理不规范、泄漏、巡检维护不到位、\n    1.7：包括导热油锅炉、压力容器、电梯、起重机械（行车、5吨以上葫芦，不包括施工吊车）相关问题。包括：安装不规范、附件缺失或失效、工程收尾不及时、检定制度未执行、泄漏、巡检维护不到位、未注册。\n    1.8：施工和检测过程中使用的机具。例如发电机、焊机、潜水泵、检测设备、热处理设备、车辆、挖机、脚手架、模板、砂轮机、电锤、电镐、梯子、线夹、混凝土振动棒的相关问题。包括：搭设不规范、附件缺失或失效、选型不当、自检制度未执行、未配备或使用。\n2、管理问题：包括目视化管理、化学品管理、消防管理、应急管理、信息化管理、作业许可管理、项目管理程序、重要现场资料管理、生产运行管理问题。具体问题包括：\n    2.1：操作规程、标识标牌、检验铭牌看不清楚、高低限、安全技术说明书、电路接线图。站内管道走向、闽门编号问题。包括：标识缺失，标识销误。\n    2.2：汽油、紫油、五剂、机油、油漆，气瓶，以及其他生产需要的药剂相关问题，包括：应急设施不完善，储存不规范，化学品超有效期，泄漏，附件款失或失效，储备不足，使用维护不到位，台账记录不规范。\n    2.3：指预防和解决火灾等灾害，包括现场救援、设备抢救、财产保护和灭火等。灭火器、消防水系统、消防沙、消防铲、喷淋系统、烟感系统、七氟丙烷灭火系统、二氧化碳灭火系统、消防通道、防火墙、防火门、应急疏散箱。相关问题包括：本质故障，系统不完善，超期未更换，通道堵塞，未配备或使用，巡检维护不到位，风险未消除。\n    2.4：为减轻事态危害、保护员工安全及恢复正常生产生活秩序而采取的各种措施和行动。包括风向标、安全门、手电筒、对讲机、救生艇、应急照明等应急装备，应急预案（应急处置卡）、应急演练（计划、演练情况、总结评价）。相关问题包括：本质故障，应急物资配备不齐全，通道堵塞，巡检维护不到位，风险未消除，应急策划不规范，未执行演练计划，应急演练执行不规范。\n    2.5：上位机、IP电话管理问题。相关问题包括：系统不完善，系统报警未处理，参数设置不合理，值号传输故障。\n    2.6：只保留了吊赖，动土，受展空间的作业管理问吗。租关问睡包括：来热行作业许可，安金抗术交底不规范，风险识别不到位，随藏执行不规起，工艺纪俄执行不到位，吊装作业不规道，动士作业不规道，要限空间作业不规范。\n    2.7：施工机具材料报审报验、项目复工报审、施工过程记录，包括取样、工程资料问题。\n    2.8：重要现场管理资料问题，指导现场生产问题。相关问题包括：未下发文件，文件编制缺项，文件编制有误。\n    2.9：主要包括生产上人员安排，准入制度执行，操作过程，人员巡检过程，记录上报，上锁，铅封方面存在的问题。包括：工艺纪律执行不到位、人员安排不合理、事故事件管理不规范、未执行操作规程、巡检要求未落实、准入制度未落实。\n3、人员能力问题：凸显人员存在问题，包括资质、培训、行为，包括：人员资质不符，培训不到位，不安全行为。\n4、环境保护：工业固废、危废管理，生活垃圾，气田水罐、药剂加注围堰（防扬散、防流失、防渗漏）相关问题。包括：应急设施不完善、储存不规范、无序排放、固废（危废）未按要求收集、储存。\n5、个人防护用品问题：个人使用的安全防护用品，不包括共同使用防护用品，不包括空呼、气体检测仪。例如安全带、护目镜、安全帽、耳塞、个人剂量计、辐射报警仪、焊工手套、焊接面罩、绝缘胶垫、绝缘靴、绝缘手套相关问题。包括：未配备或使用、使用不规范、选型不当、防护器材未检验。\n6、、施工工程问题：包括施工设计、施工方案、工程质量、工程监理问題，具体问题包括：\n    6.1：施工设计绵审不严道，设计变更执行不到位。\n    6.2：施工方案、施工预案、操作规程、焊规、焊评等施工单位的施工指导资料相关问题，包括检测单位。不包括记录、报审报验程序资料。相关问题包括：编审不严谨、风险识别不全、现场无方案\n    6.3：施工单位现场施工过程中已经造成或将要造成施工质量问题（只包括现场施工）。相关问题包括：行为质量不达标，实体质量不达标。\n    6.4：施工项目监理职责范围内存在问题，部分项目建设单位代为履行监理职责的相关问题，包括：报审报验不合规、监理未履职、未配备或使用、监理资料缺失或不全、人员越级履职、人员资质不符。\n7、作业场所问题：包括施工作业场地，包括作业平面、作业上方、通道存在的风险未消除（有风险在，把他取掉拿走就可以了，比如上面有悬挂物掉落风险，消除取掉即可）。防护措施未落实（需要增加防护的措施，一般护栏、截水沟、警戒线等）。\n\n##注意##\n1、站外管道及辅助设施的目视化问题归设施问题类，不厲于目视化管理类。\n2、采气、输气管道标识归设施问题类，不属于目视化管理类。\n3、压力表、安全阀、接地、特种设备（压力容器、行车）检定标识显示超期或无等归设施问题类，不脚于目视化问题\n4、化学品加注区、化学品遗撒造成环境污染归环境保护类，不属于管理问题类。\n5、已造成环境污染的归环境保护类。\n7、气田水罐存储属于环境保护类，站外气田水管线属于设施问题类。\n9、未光谱检查、硬度检 不属于施工工程问题类，因为有没有检测不会造成质量问题，检测只是证明有没有问题。\n10、施工工程问题类不包括记录、报审根验程序资料。\n11、工程资料相关问題属于管理类问題，不属于施工方案。\n12、黄绿标识问题归管理问题类。\n13、不戴个人防护用品问题归个人防护用品问题类。\n14、已施工不符合设计要求，焊材不一致（焊评与焊规不一致，燥规与现场不一致）提高里视程度都归在施工工程类。\n15、如果详细描述什么机具材料无质量证明文件归到设施问题类。\n16、不要输出'问题分类标签'以外的标签。\n \n##输出示例##\n输入：\"问题和隐患描述文本...\"\n输出：分类标签\n"
    user="##问题和隐患描述##\n'''\n个别食堂工作人员对灭火器的使用方法掌握不到位。\n'''\n\n##问题分类标签##\n设施问题、管理问题、人员能力问题、环境保护问题、个人防护用品问题、施工工程问题、作业场所问题\n\n##注意##\n不要输出'问题分类标签'以外的标签。\n"

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

        print(f'cost time:{round(ens - bgn, 2)}')
        # print(f'system:{system}')
        print(f'user:{user}')
        print(f'predict:{output}')

