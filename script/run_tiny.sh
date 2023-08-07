#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

save_dir=MS-Huang0714/my-tinybert_final
save_dir_model=MS-Huang0714/my-tinybert_model_final
mkdir -p ${save_dir}
mkdir -p ${save_dir_model}

# budget_flops=4.5e6
budget_flops=4.5e6
max_layers=4
input_image_size=224
population_size=128
evolution_max_iter=1000000

export CUDA_VISIBLE_DEVICES=1

python evolutionary_search.py --gpu 0 \
  --zero_shot_score TPC_fast \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size ${input_image_size} \
  --num_classes 1000 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}

# python -m torch.distributed.launch --nproc_per_node=16 train_model.py --lm_type=mlm --dataset_id="olm/olm-october-2022-tokenized-512" \
#     --repository_id=${save_dir} --tokenizer_id="muhtasham/olm-bert-tiny-december-2022" \
#     --model_config_id="muhtasham/olm-bert-tiny-december-2022" --adam_beta2=0.98 --adam_epsilon=1e-6 --adam_beta1=0.9 --warmup_steps=24000 \
#     --max_steps=100000 --per_device_train_batch_size=20 --gradient_accumulation_steps=25 --learning_rate=6e-4

python -m torch.distributed.launch --nproc_per_node=1 train_model.py --lm_type=mlm --dataset_id="olm/olm-october-2022-tokenized-512" \
    --load_model_dir=${save_dir} --repository_id=${save_dir_model} --tokenizer_id="olm-bert-tiny-december-2022" \
    --model_config_id="olm-bert-tiny-december-2022" --adam_beta2=0.98 --adam_epsilon=1e-6 --adam_beta1=0.9 --warmup_steps=24000 \
    --max_steps=100000 --per_device_train_batch_size=18 --gradient_accumulation_steps=28 --learning_rate=6e-4

