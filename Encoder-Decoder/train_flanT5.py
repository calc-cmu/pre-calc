import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import random

def get_num(sent):
    # also check for decimals and fractions
    res = [i for i in sent.split() if i.isdigit() or '/' in i or '.' in i]
    return res[0]

def get_sent_pairs(row):

    input_sent = row['question']
    if 'how' in input_sent:
        sent1 = input_sent.split('how')[0]
    elif 'How' in input_sent:
        sent1 = input_sent.split('How')[0]
    else:
        sent1 = input_sent

    sent2_ent = row['ent_outputs'].split('[/INST]')[-1].split('[END]')[1].split('Output:')[-1]
    sent2_ctr = row['ctr_outputs'].split('[/INST]')[-1].split('[END]')[1].split('Output:')[-1]

    ent_num = get_num(sent2_ent)
    ctr_num = get_num(sent2_ctr)

    equation = row['equation'].split('=')
    if 'x' in equation[0]:
      equation = equation[1].strip()
    else:
      equation = equation[0].strip()

    return sent1, sent2_ent, sent2_ctr, ent_num, ctr_num, equation

def seq2seq_input_output(sent1, sent2, equation,num):

    input = f"nli sentence1: {sent1} sentence2: {sent2}"
    output = f"<equate> ({equation}, {num})"

    return input, output

def get_all_ip_op(df):

    inputs = []
    outputs = []

    for i in range(df.shape[0]):
      sent1, sent2_ent, sent2_ctr, ent_num, ctr_num, equation = get_sent_pairs(df.iloc[i])
      ip, op = seq2seq_input_output(sent1, sent2_ent, equation,ent_num)
      inputs.append(ip)
      outputs.append(op)

      ip, op = seq2seq_input_output(sent1, sent2_ctr, equation,ctr_num)
      inputs.append(ip)
      outputs.append(op)

    return inputs, outputs

train_df = pd.read_csv('data/train_preft_flant5.csv')
test_df = pd.read_csv('data/test_preft_flant5.csv')
print(train_df.shape)
print(test_df.shape)

columns = train_df.columns.tolist()

train_df = Dataset.from_pandas(train_df)
test_df = Dataset.from_pandas(test_df)

## Tokenization

model_id="google/flan-t5-large"
# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([train_df,test_df]).map(lambda x: tokenizer(x["input_prompt"], truncation=True), batched=True, remove_columns=columns)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([train_df,test_df]).map(lambda x: tokenizer(x["output_expression"], truncation=True), batched=True, remove_columns=columns)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")


def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["input_prompt"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output_expression"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_df.map(preprocess_function, batched=True, remove_columns=columns)
tokenized_test = test_df.map(preprocess_function, batched=True, remove_columns=columns)
print(f"Keys of tokenized dataset: {list(tokenized_train.features)}")

nltk.download("punkt")
# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

model = AutoModelForSeq2SeqLM.from_pretrained(model_id,device_map="cuda:1")

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

repo_name = "flan-t5-large-mawpnli-calcx-nli-pt"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repo_name,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{repo_name}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    # save_strategy="no",
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()
tokenizer.push_to_hub()