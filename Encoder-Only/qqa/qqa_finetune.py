from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice,
    AutoModel,
    Trainer, 
    TrainingArguments
)
import numpy as np
from datasets import Dataset, load_dataset
import evaluate
import accelerate
import argparse
from qqa_utils import preprocess_qqa, DataCollatorForQQA
from torch import nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='bert-base-uncased', 
                    choices=[
                                'bert-base-uncased',
                                'roberta-base',
                                'michiyasunaga/LinkBERT-base',
                                'ProsusAI/finbert',
                                'vishwa27/CN_BERT_Sci',
                                'vishwa27/CN_BERT_Digit',
                                'Kghate/CN_RoBERTa_Dig',
                                'vishruthnath/CN_RoBERTa_Sci',
                                'vishruthnath/Calc_new_BERT_ep20',
                                'vishruthnath/Calc_new_RoBERTa_ep20'
                            ])
parser.add_argument('--version', default='ORG', choices=['ORG', 'DIG', 'SCI'])
parser.add_argument('--reset_classifier', action='store_true', default=False)
args = parser.parse_args()

# Get data
train_file = f'QQA_train.json'
dev_file = f'QQA_dev.json'
test_file = f'QQA_test.json'
qqa = load_dataset("json", data_files={"train": train_file, "validation": dev_file, "test": test_file}) 

# Pick model 
# model_id = 'bert-base-uncased'
# model_id = 'roberta-base'
# model_id = 'michiyasunaga/LinkBERT-base'
# model_id = 'ProsusAI/finbert'

# MOdel and tokenizer
model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMultipleChoice.from_pretrained(model_id, ignore_mismatched_sizes=True, num_labels=2)
# model.classifier = nn.Linear(model.base_model.config.hidden_size, 2) # TODO fix resetting head 
if args.reset_classifier:
    print('Resetting classifier weights')
    model.classifier.apply(model._init_weights) # TODO TEST THIS

# hf_model = AutoModel.from_pretrained(model_id)
# print(hf_model.base_model.config.hidden_size)
# model = nn.Sequential(
#         hf_model.base_model,
#         nn.Linear(hf_model.base_model.config.hidden_size, 2)
# )


# Preprocess Data
tokenized_qqa = qqa.map(preprocess_qqa, batched=True, batch_size= 8, 
                            fn_kwargs={'tokenizer': tokenizer, 'version': args.version}
                        )

# Evaluate
accuracy = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
F1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
  
    # return accuracy.compute(predictions=predictions, references=labels)
    return {
        "f1": F1_metric.compute(predictions=predictions, references=labels),
        "accuracy": accuracy.compute(predictions=predictions, references=labels)
    }


# Train
training_args = TrainingArguments(
    output_dir=f"QQA_{model_id.split('/')[-1]}_{args.version}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=1e-05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    # weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_qqa["train"],
    eval_dataset=tokenized_qqa["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForQQA(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
