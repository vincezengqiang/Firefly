CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    --train_args_file /root/autodl-tmp/a00_Firefly/train_args/sft_mine/qwen-7b-sft-full.json