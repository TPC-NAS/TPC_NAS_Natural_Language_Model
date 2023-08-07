import math

# from ..timm.models.davit import DaViT_tiny, _create_transformer

def compute_nas_score(random_structure_str, num_classes):
    depth        = random_structure_str["depth"]
    input_channel = random_structure_str["input_channel"]
    SelfAttention_out_channel = random_structure_str["SelfAttention_out_channel"]
    BertIntermediate_out_channel = random_structure_str["BertIntermediate_out_channel"]
    num_of_heads = random_structure_str["num_of_heads"]

    nas_score = 0.0

    # main_block params
    for i in range(depth):
        tmp_input_channel = input_channel
        # main_blocks_size += input_channel * SelfAttention_out_channel[i] * 3
        nas_score += math.log2(SelfAttention_out_channel[i])*3

        input_channel = SelfAttention_out_channel[i]
        # main_blocks_size += input_channel * tmp_input_channel
        nas_score += math.log2(tmp_input_channel)

        input_channel = tmp_input_channel
        # main_blocks_size += input_channel * BertIntermediate_out_channel[i]
        nas_score += math.log2(BertIntermediate_out_channel[i])

        input_channel = BertIntermediate_out_channel[i]
        # main_blocks_size += input_channel * tmp_input_channel
        nas_score += math.log2(tmp_input_channel)

        input_channel = tmp_input_channel

    return nas_score

if __name__ == "__main__":
    model_kwargs = dict(depth=12, input_channel=768, SelfAttention_out_channel=[768]*12,
                                BertIntermediate_out_channel=[3072]*12, num_of_heads=[12, 12, 12,12])

    # the_model = _create_transformer('DaViT_224', pretrained=False, **model_kwargs)
    print(compute_nas_score(model_kwargs, 1000))
