from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from transformers import (
    HfArgumentParser,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    set_seed,
    AutoConfig,
    DataCollatorForLanguageModeling,
)

import torch.nn as nn
import torch
import torch.nn.functional as F

dataset = "wnli"

if dataset == "mrpc":
    raw_datasets = load_dataset("glue", "mrpc")
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    num_labels = 2
    eval_name = "validation"
    # teacher_id = "textattack/roberta-base-MRPC"

elif dataset == "mnli":
    raw_datasets = load_dataset("glue", "mnli")
    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)
    num_labels = 3
    eval_name = "validation_mismatched"
    # eval_name = "validation_matched"
    # teacher_id = "textattack/roberta-base-MNLI"

elif dataset == "qnli":
    raw_datasets = load_dataset("glue", "qnli")
    def tokenize_function(example):
        return tokenizer(example["question"], example["sentence"], truncation=True)
    num_labels = 2
    eval_name = "validation"
    # teacher_id = "textattack/roberta-base-QNLI"

elif dataset == "sst2":
    raw_datasets = load_dataset("glue", "sst2")
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)
    num_labels = 2
    eval_name = "validation"
    # teacher_id = "textattack/roberta-base-SST-2"

elif dataset == "qqp":
    raw_datasets = load_dataset("glue", "qqp")
    def tokenize_function(example):
        return tokenizer(example["question1"], example["question2"], truncation=True)
    num_labels = 2
    eval_name = "validation"
    # teacher_id = "textattack/roberta-base-QQP"

elif dataset == "rte":
    raw_datasets = load_dataset("glue", "rte")
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    num_labels = 2
    eval_name = "validation"
    # teacher_id = "textattack/roberta-base-RTE"

elif dataset == "wnli":
    raw_datasets = load_dataset("glue", "wnli")
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    num_labels = 2
    eval_name = "validation"
    # teacher_id = "textattack/roberta-base-WNLI"

teacher_id = "olm/olm-roberta-base-dec-2022"
# teacher_id = "xlm-roberta-large"

# checkpoint = "olm-bert-tiny-december-2022"
checkpoint = "/home/w771/huang/olm-training/MS-Huang0714/my-roberta-based_model/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# print(tokenized_datasets["train"].features.keys())
labels = tokenized_datasets["train"].features["label"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
config = AutoConfig.from_pretrained(checkpoint)

print(config)

import json
path = "/home/w771/huang/olm-training/MS-Huang0714/my-roberta-based_final/best_structure.txt"
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
config.hidden_size = input_channel
config.num_labels=num_labels
config.id2label=id2label
config.label2id=label2id
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

model_type               = config.model_type

model = AutoModelForSequenceClassification.from_config(config)

model.resize_token_embeddings(len(tokenizer))

def replace_layers(model, key, index, old, new, father_name):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            # compound module, go inside it
            replace_layers(module, key, index, old, new, father_name+"."+n)

        if isinstance(module, old):
            new_father_name = father_name + "." + n
            if key in new_father_name and index in new_father_name:
                setattr(model, n, new)

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

state_dict = torch.load("/home/w771/huang/olm-training/MS-Huang0714/my-roberta-based_model/checkpoint-97500/pytorch_model.bin")
# print(state_dict)

model.load_state_dict(state_dict, strict=False)
# model.encoder.layer.0.
# print(model)

from transformers import Trainer

# predictions = trainer.predict(tokenized_datasets[eval_name])
# print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np

# preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

# metric = evaluate.load("glue", "mrpc")
# metric.compute(predictions=preds, references=predictions.label_ids)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", dataset)
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", per_device_train_batch_size=16)
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", per_device_train_batch_size=28, num_train_epochs=10, fp16=True, save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="accuracy", learning_rate=1e-5, max_grad_norm=0.05)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# teacher model
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
# teacher_model.resize_token_embeddings(len(tokenizer))
assert model.config.vocab_size == teacher_model.config.vocab_size
assert model.config.hidden_size == teacher_model.config.hidden_size
assert model.config.max_position_embeddings == teacher_model.config.max_position_embeddings

sample = "This is a basic example, with different words to test."
print(teacher_tokenizer(sample))
print(tokenizer(sample))
assert teacher_tokenizer(sample) == tokenizer(sample), "Tokenizers haven't created the same output"

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

from huggingface_hub import HfFolder

repo_name = "evaluation"

# training_args = DistillationTrainingArguments(
#     # "test-trainer",
#     output_dir=repo_name,
#     num_train_epochs=5,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=128,
#     fp16=True,
#     learning_rate=6e-5,
#     seed=33,
#     # logging & evaluation strategies
#     logging_dir=f"{repo_name}/logs",
#     logging_strategy="epoch", # to get more information to TB
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     report_to="tensorboard",
#     # push to hub parameters
#     push_to_hub=False,
#     # hub_strategy="every_save",
#     # hub_model_id=repo_name,
#     # hub_token=HfFolder.get_token(),
#     # distilation parameters
#     alpha=0.5,
#     temperature=4.0
#     )

# trainer = Trainer(
#     teacher_model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets[eval_name],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# trainer = DistillationTrainer(
#     model,
#     training_args,
#     teacher_model=teacher_model,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets[eval_name],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
