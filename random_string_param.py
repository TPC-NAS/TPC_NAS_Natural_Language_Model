from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
import torch
import torch.nn as nn

from transformers import (
    HfArgumentParser,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    set_seed,
    AutoConfig,
    DataCollatorForLanguageModeling,
)

from transformers import Trainer, TrainingArguments, AutoTokenizer

def get_num_layers(random_structure_str): # no use
    depths = sum(random_structure_str["depths"])

    return depths

def get_model_size(random_structure_str, num_classes): # only count encoder
    patch_embeds_size = 0
    main_blocks_size  = 0
    head_size         = 0
    norm_size         = 0

    depth        = random_structure_str["depth"]
    input_channel = random_structure_str["input_channel"]
    SelfAttention_out_channel = random_structure_str["SelfAttention_out_channel"]
    BertIntermediate_out_channel = random_structure_str["BertIntermediate_out_channel"]
    num_of_heads = random_structure_str["num_of_heads"]

    # main_block params
    for i in range(depth):
        tmp_input_channel = input_channel
        main_blocks_size += input_channel * SelfAttention_out_channel[i] * 3
        # replace_layers(model, "query", str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        # replace_layers(model, "key"  , str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        # replace_layers(model, "value", str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        input_channel = SelfAttention_out_channel[i]
        main_blocks_size += input_channel * tmp_input_channel
        main_blocks_size += tmp_input_channel * 2
        # replace_layers(model, "attention.output", str(i), nn.Linear, nn.Linear(input_channel, tmp_input_channel), "")
        # replace_layers(model, "attention.output", str(i), nn.LayerNorm, nn.LayerNorm(tmp_input_channel), "")
        input_channel = tmp_input_channel
        main_blocks_size += input_channel * BertIntermediate_out_channel[i]
        # replace_layers(model, "intermediate.dense", str(i), nn.Linear, nn.Linear(input_channel, BertIntermediate_out_channel[i]), "")
        input_channel = BertIntermediate_out_channel[i]
        main_blocks_size += input_channel * tmp_input_channel
        main_blocks_size += tmp_input_channel * 2
        # replace_layers(model, str(i)+".output.dense", str(i), nn.Linear, nn.Linear(input_channel, tmp_input_channel), "")
        # replace_layers(model, str(i)+".output.LayerNorm", str(i), nn.LayerNorm, nn.LayerNorm(tmp_input_channel), "")
        input_channel = tmp_input_channel

    model_size = main_blocks_size

    # print("patch_embeds_size = ", patch_embeds_size)
    # print("main_blocks_size  = ", main_blocks_size)
    # print("head_size = ", head_size)

    return model_size

def get_FLOPs(random_structure_str, num_classes): # only count encoder
    main_blocks_flop  = 0

    depth        = random_structure_str["depth"]
    input_channel = random_structure_str["input_channel"]
    SelfAttention_out_channel = random_structure_str["SelfAttention_out_channel"]
    BertIntermediate_out_channel = random_structure_str["BertIntermediate_out_channel"]
    num_of_heads = random_structure_str["num_of_heads"]

    # main_block params
    for i in range(depth):
        tmp_input_channel = input_channel
        main_blocks_flop += input_channel * SelfAttention_out_channel[i] * 3 * 6
        # replace_layers(model, "query", str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        # replace_layers(model, "key"  , str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        # replace_layers(model, "value", str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        input_channel = SelfAttention_out_channel[i]
        main_blocks_flop += input_channel * tmp_input_channel * 6
        # replace_layers(model, "attention.output", str(i), nn.Linear, nn.Linear(input_channel, tmp_input_channel), "")
        # replace_layers(model, "attention.output", str(i), nn.LayerNorm, nn.LayerNorm(tmp_input_channel), "")
        input_channel = tmp_input_channel
        main_blocks_flop += input_channel * BertIntermediate_out_channel[i] * 6
        # replace_layers(model, "intermediate.dense", str(i), nn.Linear, nn.Linear(input_channel, BertIntermediate_out_channel[i]), "")
        input_channel = BertIntermediate_out_channel[i]
        main_blocks_flop += input_channel * tmp_input_channel * 6
        # replace_layers(model, str(i)+".output.dense", str(i), nn.Linear, nn.Linear(input_channel, tmp_input_channel), "")
        # replace_layers(model, str(i)+".output.LayerNorm", str(i), nn.LayerNorm, nn.LayerNorm(tmp_input_channel), "")
        input_channel = tmp_input_channel

        # matrix multiplication
        # print(2 * SelfAttention_out_channel[i] * input_channel)
        # main_blocks_flop += 2 * SelfAttention_out_channel[i]**2

    model_flops = main_blocks_flop

    return model_flops


if __name__ == '__main__':
    # tinyBERT
    model_kwargs = dict(depth=12, input_channel=768, SelfAttention_out_channel=[768]*12,
                                BertIntermediate_out_channel=[3072]*12, num_of_heads=[12, 12, 12,12])

    vocab_size = 50265
    print("model size = ",  get_model_size(model_kwargs, vocab_size))
    print("model flops = ", get_FLOPs(model_kwargs, vocab_size))

    # build model
    # config = AutoConfig.from_pretrained("olm-bert-tiny-december-2022")
    config = AutoConfig.from_pretrained("roberta-base")
    model  = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(vocab_size)

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 0], [0, 0, 0]])

    input_data = (input_ids, input_mask, token_type_ids)

    flops = FlopCountAnalysis(model, input_data)
    print("print original model")
    # flop_count_table(flops)
    # print("true flops = ", flops.total())
    # print(flop_count_str(flops))
    print(flop_count_table(flops))

    depth = model_kwargs["depth"] # BertLayer: BertSelfAttention -> BertSelfOutput -> BertIntermediate -> BertOutput
    input_channel = model_kwargs["input_channel"]
    SelfAttention_out_channel = model_kwargs["SelfAttention_out_channel"]
    # SelfOutput_out_channel = [64, 128, 256, 512] # do not change size due to the skip connection
    BertIntermediate_out_channel = model_kwargs["BertIntermediate_out_channel"]
    # BertOutput_out_channel = [64, 128, 256, 512] # do not change size due to the skip connection
    num_of_heads = model_kwargs["num_of_heads"]

    def replace_layers(model, key, index, old, new, father_name):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                # compound module, go inside it
                replace_layers(module, key, index, old, new, father_name+"."+n)

            if isinstance(module, old):
                new_father_name = father_name + "." + n
                if key in new_father_name and index in new_father_name:
                    setattr(model, n, new)

    # print(model)

    for i in range(depth):
        tmp_input_channel = input_channel
        replace_layers(model, "query", str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        replace_layers(model, "key"  , str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        replace_layers(model, "value", str(i), nn.Linear, nn.Linear(input_channel, SelfAttention_out_channel[i]), "")
        input_channel = SelfAttention_out_channel[i]
        replace_layers(model, "attention.output", str(i), nn.Linear, nn.Linear(input_channel, tmp_input_channel), "")
        replace_layers(model, "attention.output", str(i), nn.LayerNorm, nn.LayerNorm(tmp_input_channel), "")
        input_channel = tmp_input_channel
        replace_layers(model, "intermediate.dense", str(i), nn.Linear, nn.Linear(input_channel, BertIntermediate_out_channel[i]), "")
        input_channel = BertIntermediate_out_channel[i]
        replace_layers(model, str(i)+".output.dense", str(i), nn.Linear, nn.Linear(input_channel, tmp_input_channel), "")
        replace_layers(model, str(i)+".output.LayerNorm", str(i), nn.LayerNorm, nn.LayerNorm(tmp_input_channel), "")
        input_channel = tmp_input_channel

    for name, module in model.named_modules():
        for i in range(depth):
            self_attention_str = "bert.encoder.layer." + str(i) + ".attention.self"
            if name == self_attention_str:
                setattr(module, "attention_head_size", int(SelfAttention_out_channel[i]/num_of_heads[i]))
                setattr(module, "all_head_size", int(SelfAttention_out_channel[i]))
                setattr(module, "num_attention_heads", int(num_of_heads[i]))

    # print(model)

    # calculate FLOPs and Params
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 0], [0, 0, 0]])

    input_data = (input_ids, input_mask, token_type_ids)

    flops = FlopCountAnalysis(model, input_data)
    # flop_count_table(flops)
    # print("true flops = ", flops.total())
    # print(flop_count_str(flops))
    print(flop_count_table(flops))

