import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from t5_data_collator import DataCollatorForT5MLM, compute_t5_input_and_target_lengths

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
from datasets import load_dataset
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
# from TinyBert.transformer import *

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments which aren't included in the TrainingArguments
    """

    dataset_id: str = field(
        default=None, metadata={"help": "The repository id of the dataset to use (via the datasets library)."}
    )
    tokenizer_id: str = field(
        default=None, metadata={"help": "The repository id of the tokenizer to use (via AutoTokenizer)."}
    )
    repository_id: str = field(
        default=None,
        metadata={"help": "The repository id where the model will be saved or loaded from for futher pre-training."},
    )
    load_model_dir: str = field(
        default=None,
        metadata={"help": "Place to load model config."},
    )
    hf_hub_token: str = field(
        default=True,
        metadata={"help": "The Token used to push models, metrics and logs to the Hub."},
    )
    lm_type: str = field(
        default=None,
        metadata={"help": "The type of language model to train. Options are mlm, clm, or t5. t5 is a WIP for now and may not work as expected."},
    )
    model_config_id: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The Batch Size per HPU used during training, default:16"},
    )
    max_steps: Optional[int] = field(
        default=460000,
        metadata={"help": "The Number of Training steps to perform."},
    )
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "Learning Rate for the training"})
    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mean_noise_span_length: Optional[float] = field(
        default=3.0, metadata={"help": "Mean span length of masked tokens for T5."}
    )
    desired_t5_input_length: Optional[int] = field(
        default=512, metadata={"help": "The desired number of tokens for T5 input. Because T5 masking alters the number of tokens, this value must be specified."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "Number of gradient accumulation steps to take; artificially increases the batch size"}
    )
    warmup_steps: Optional[int] = field(
        default=10000, metadata={"help": "Number of learning warmup steps to take"}
    )
    adam_beta1: Optional[float] = field(
        default=0.9, metadata={"help": "Parameter for the adam optimizer"}
    )
    adam_beta2: Optional[float] = field(
        default=0.999, metadata={"help": "Parameter for the adam optimizer"}
    )
    adam_epsilon: Optional[float] = field(
        default=1e-6, metadata={"help": "Parameter for the adam optimizer"}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Parameter for the adam optimizer. Regularization to prevent weights from getting too big."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "LR scheduler type, such as cosine or linear."}
    )
    fp16: Optional[bool] = field(
        default=True, metadata={"help": "fp16"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "gradient_checkpointing"}
    )
    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config if using deepspeed"})


def train_model():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Script parameters {script_args}")

    # set seed for reproducibility
    seed = 123
    set_seed(seed)

    # define our hyperparameters
    training_args = TrainingArguments(
        output_dir=script_args.repository_id,
        local_rank=script_args.local_rank,
        deepspeed=script_args.deepspeed,
        # logging & evaluation strategies
        logging_dir=f"{script_args.repository_id}/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=2500,
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=script_args.repository_id,
        hub_token=script_args.hf_hub_token,
        # optimization parameters
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        learning_rate=script_args.learning_rate,
        seed=seed,
        max_steps=script_args.max_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        warmup_steps=script_args.warmup_steps,
        adam_beta1=script_args.adam_beta1,
        adam_beta2=script_args.adam_beta2,
        bf16 = True,
        adam_epsilon=script_args.adam_epsilon,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,

        # fp16=script_args.fp16,
        # gradient_checkpointing=script_args.gradient_checkpointing,
    )

    # load processed dataset
    train_dataset = load_dataset(script_args.dataset_id, split="train")

    # load trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id, use_auth_token=script_args.hf_hub_token)

    # load config (for training from scratch, we call .from_config())
    logger.info("Training new model from scratch")
    config = AutoConfig.from_pretrained(script_args.model_config_id)

    print(config)

    import json
    path = script_args.load_model_dir + "/best_structure.txt"
    model_kwargs = json.load(open(path))
    print(model_kwargs)

    depth = model_kwargs["depth"] # BertLayer: BertSelfAttention -> BertSelfOutput -> BertIntermediate -> BertOutput
    input_channel = model_kwargs["input_channel"]
    SelfAttention_out_channel = model_kwargs["SelfAttention_out_channel"]
    # SelfOutput_out_channel = [64, 128, 256, 512] # do not change size due to the skip connection
    BertIntermediate_out_channel = model_kwargs["BertIntermediate_out_channel"]
    # BertOutput_out_channel = [64, 128, 256, 512] # do not change size due to the skip connection
    num_of_heads = model_kwargs["num_of_heads"]

    config.num_hidden_layers = depth
    config.hidden_size       = input_channel
    model_type               = config.model_type

    # Set up the model and data collator.
    if script_args.lm_type == "mlm":
        model = AutoModelForMaskedLM.from_config(config)
        # model = TinyBertForPreTraining.from_scratch(script_args.model_config_id)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=script_args.mlm_probability, pad_to_multiple_of=8
        )
    elif script_args.lm_type == "clm":
        model = AutoModelForCausalLM.from_config(config)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )
    elif script_args.lm_type == "t5":
        train_dataset = train_dataset.remove_columns(["attention_mask", "special_tokens_mask"])
        input_length = script_args.desired_t5_input_length
        expanded_inputs_length, target_length = compute_t5_input_and_target_lengths(
            inputs_length=input_length,
            noise_density=script_args.mlm_probability,
            mean_noise_span_length=script_args.mean_noise_span_length,
        )
        assert expanded_inputs_length == len(train_dataset[0]["input_ids"]),\
            f"""
            You have specified that the T5 input length should be {script_args.desired_t5_input_length}.
            In order to do this, the examples in the dataset need to be {expanded_inputs_length} before masking.
            But the examples in the dataset actually appear to be {len(train_dataset[0]['input_ids'])} tokens long.
            """
        model = T5ForConditionalGeneration._from_config(config)
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=script_args.mlm_probability,
            mean_noise_span_length=script_args.mean_noise_span_length,
            input_length=input_length,
            target_length=target_length,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
    )
    else:
        raise ValueError("Unrecognized lm_type. Options are mlm, clm, or t5.")

    logger.info(f"Resizing token embedding to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # change depth in config
    # only change the channel size in the encoder

    # print(model.bert)

    def replace_layers(model, key, index, old, new, father_name):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                # compound module, go inside it
                replace_layers(module, key, index, old, new, father_name+"."+n)

            if isinstance(module, old):
                new_father_name = father_name + "." + n
                if key in new_father_name and index in new_father_name:
                    setattr(model, n, new)

    # embedding
    # replace_layers(model, "word_embeddings", "embeddings", nn.Embedding, nn.Embedding(config.vocab_size, input_channel, padding_idx=0), "")
    # replace_layers(model, "position_embeddings", "embeddings", nn.Embedding, nn.Embedding(config.max_position_embeddings, input_channel), "")
    # replace_layers(model, "token_type_embeddings",  "embeddings", nn.Embedding, nn.Embedding(config.type_vocab_size, input_channel), "")
    print("model_type = ", model_type)
    if model_type == "roberta":
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
                self_attention_str = "roberta.encoder.layer." + str(i) + ".attention.self"
                if name == self_attention_str:
                    print("hello")
                    setattr(module, "attention_head_size", int(SelfAttention_out_channel[i]/num_of_heads[i]))
                    setattr(module, "all_head_size", int(SelfAttention_out_channel[i]))
                    setattr(module, "num_attention_heads", int(num_of_heads[i]))

    else: # bert
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

    print(model)

    # state_dict = torch.load("/home/w771/huang/olm-training/MS-Huang0714/my-bert-based_model/checkpoint-60000/pytorch_model.bin")
    # model.load_state_dict(state_dict, strict=False)

    # optimizer_state_dict = torch.load("/home/w771/huang/olm-training/MS-Huang0714/my-bert-based_model/checkpoint-60000/pytorch_model.bin")
    # model.load_state_dict(state_dict, strict=False)

    # calculate FLOPs and Params
    input_ids = torch.LongTensor([[590, 0, 0], [15, 5, 0]])
    input_mask = torch.LongTensor([[2, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 0], [0, 0, 0]])

    input_data = (input_ids, input_mask, token_type_ids)

    flops = FlopCountAnalysis(model, input_data)
    print(flop_count_table(flops))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train the model
    trainer.train(script_args.resume_from_checkpoint)
    # trainer.train("checkpoint-60000")


if __name__ == "__main__":
    train_model()
