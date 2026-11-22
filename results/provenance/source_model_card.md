---
license: apache-2.0
base_model: WinKawaks/vit-small-patch16-224
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: vit-small-patch16-224-finetuned-Cifar10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# vit-small-patch16-224-finetuned-Cifar10

This model is a fine-tuned version of [WinKawaks/vit-small-patch16-224](https://huggingface.co/WinKawaks/vit-small-patch16-224) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0378
- Accuracy: 0.9876

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.3129        | 1.0   | 351  | 0.0535          | 0.9818   |
| 0.2333        | 2.0   | 703  | 0.0492          | 0.9836   |
| 0.1991        | 2.99  | 1053 | 0.0378          | 0.9876   |


### Framework versions

- Transformers 4.38.2
- Pytorch 2.2.1+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2
