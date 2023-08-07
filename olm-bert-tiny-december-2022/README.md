---
datasets:
- olm/olm-december-2022-tokenized-512
widget:
  - text: The Covid-19 <mask> is global
language:
- en
pipeline_tag: fill-mask
license: apache-2.0
---


# Tiny BERT December 2022

This is a more up-to-date version of the [original tiny BERT](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962) (English only, uncased, trained with WordPiece masking).
In addition to being more up-to-date, it is more CPU friendly than its base version, but its first version and is not perfect by no means. Took a day and 8x A100s to train. 🤗 



The model was trained on a cleaned December 2022 snapshot of Common Crawl and Wikipedia.

This model was intended to be  part of the OLM project, which has the goal of continuously training and releasing models that are up-to-date and comparable in standard language model performance to their static counterparts.
This is important because we want our models to know about events like COVID or 
a presidential election right after they happen.

## Intended uses

You can use the raw model for masked language modeling, but it's mostly intended to
be fine-tuned on a downstream task, such as sequence classification, token classification or question answering.

## Special note

It looks like the olm tinybert is underperforming the original from a quick glue finetuning and dev evaluation:

Original
```bash
{'cola_mcc': 0.0, 'sst2_acc': 0.7981651376146789, 'mrpc_acc': 0.6838235294117647, 'mrpc_f1': 0.8122270742358079, 'stsb_pear': 0.67208
2873279731, 'stsb_spear': 0.6933378278505834, 'qqp_acc': 0.7766420762598881, 'mnli_acc': 0.6542027508914926, 'mnli_acc_mm': 0.6670056
956875509, 'qnli_acc': 0.774665934468241, 'rte_acc': 0.5776173285198556, 'wnli_acc': 0.49295774647887325}
```

OLM
```bash
{'cola_mcc': 0.0, 'sst2_acc': 0.7970183486238532, 'mrpc_acc': 0.6838235294117647, 'mrpc_f1': 0.8122270742358079, 'stsb_pear': -0.1597
8233085015087, 'stsb_spear': -0.13638650127051932, 'qqp_acc': 0.6292213609628794, 'mnli_acc': 0.5323484462557311, 'mnli_acc_mm': 0.54
65825874694874, 'qnli_acc': 0.6199890170236134, 'rte_acc': 0.5595667870036101, 'wnli_acc': 0.5352112676056338}
```

Probably messed up with hyperparameters and tokenizer a bit, unfortunately. Anyway Stay tuned for version 2 🚀🚀🚀
But please try it out on your downstream tasks, might be more performant. Should be cheap to fine-tune due to its size 🤗

## Dataset

The model and tokenizer were trained with this [December 2022 cleaned Common Crawl dataset](https://huggingface.co/datasets/olm/olm-CC-MAIN-2022-49-sampling-ratio-olm-0.15114822547) plus this [December 2022 cleaned Wikipedia dataset](https://huggingface.co/datasets/olm/olm-wikipedia-20221220).\
The tokenized version of these concatenated datasets is [here](https://huggingface.co/datasets/olm/olm-december-2022-tokenized-512).\
The datasets were created with this [repo](https://github.com/huggingface/olm-datasets).

## Training

The model was trained according to the OLM BERT/RoBERTa instructions at this [repo](https://github.com/huggingface/olm-training).