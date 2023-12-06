import datetime

import datasets
import evaluate
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, DataCollatorForTokenClassification,
                          DataCollatorWithPadding)


def preprocess_mawps(example, tokenizer):
    # question = example['question_split'] + ' [OP]'
    question = example['question_split']
    question.append('[OP]')
    # print('Question', question)

    op_tags = example['operand_tags']   
    op_tags.append(-100) # Appending -100 for the OP token 
    # print('OP tags', op_tags)
    assert len(question) == len(op_tags)

    tokenized_inputs = tokenizer(question, is_split_into_words=True, add_special_tokens=True, truncation=True, return_length=True)
    # Returning length as need to find the rep for last [OP] token

    # labels = []
    # for i, label in enumerate(op_tags):
    # word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    # print('Input_ids', tokenized_inputs.input_ids)
    # print('Word_ids', word_ids, len(word_ids))

    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(op_tags[word_idx])
            previous_word_idx = word_idx
        else:
            label_ids.append(op_tags[previous_word_idx])
            # label_ids.append(-100) # Making all subsequent things -100
        # previous_word_idx = word_idx
    # labels.append(label_ids)

    # print('Labels', label_ids)

    operation = example['operation']
    tokenized_inputs['operation_labels'] = operation_label2id[operation]

    tokenized_inputs["labels"] = label_ids
    assert len(tokenized_inputs['labels']) == len(tokenized_inputs['input_ids'])
    return tokenized_inputs

    # print(example['operand_tags'])
    # op_tags = example['operand_tags']
    # op_tags.append(-100) # Appending -100 for the OP token 
    # tokenized_inputs['labels'] = op_tags
    # print('Input Ids', len(tokenized_inputs['input_ids']))
    # print('Labels', len(tokenized_inputs['labels']))

    # assert len(tokenized_inputs['labels']) == len(tokenized_inputs['input_ids'])
    # return tokenized_inputs

def process_predictions(predictions,labels):
    predictions=predictions.detach().cpu().clone().numpy()
    labels=labels.detach().cpu().clone().numpy()

    # Only considering loss for tokens where prediction_token = -1
    true_labels = [[token_id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [[token_id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    return true_labels, true_predictions


model_id = 'bert-base-uncased'
# model_id = 'roberta'

tokenizer = AutoTokenizer.from_pretrained(model_id)
special_tokens = {'additional_special_tokens': ["[OP]"]}
tokenizer.add_special_tokens(special_tokens)

# Id2label
operation_label2id = {
    '+': 0, 
    '-': 1,
    '*': 2,
    '/': 3
}
operation_id2label = {v: k for k,v in operation_label2id.items()}

token_id2label = {
    0: 'Not_Operand',
    1: 'Operand'
}
token_label2id = {v: k for k,v in token_id2label.items()}


dataset = datasets.load_dataset('vishruthnath/Calc-MAWPS-CalcBERT-Tags')
combined_tokenized_dataset = dataset.map(preprocess_mawps,
                                         fn_kwargs={"tokenizer": tokenizer},
                                         remove_columns=['id', 'chain', 'equation', 'expression', 'num_unique_ops', 'operand_tags', 'operands', 'operation', 'question', 'question_split', 'valid', '__index_level_0__', 'result'])
# kept columns (id, result, 'input_ids', 'token_type_ids', 'attention_mask', 'operation_labels', 'labels')
# rename result_float to result 
combined_tokenized_dataset.rename_column('result_float', 'result')

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

train_dataset = combined_tokenized_dataset['train']
val_dataset = combined_tokenized_dataset['validation']
test_dataset = combined_tokenized_dataset['test']

train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=8)
val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=8)
test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=8)

model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        id2label=token_id2label,
        label2id=token_label2id,
        ignore_mismatched_sizes=True
    )
model.resize_token_embeddings(len(tokenizer)) # Resize embedding to include special token [OP]


# =============== Training loop
optimizer = AdamW(model.parameters(), lr=2e-5)

num_train_epochs = 5
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

operation_head = torch.nn.Linear(model.config.hidden_size, len(operation_id2label))

token_metric = evaluate.load("seqeval")
operation_metric = evaluate.load("accuracy")

progress_bar = tqdm(range(num_training_steps))
# output_dir = '/content/drive/MyDrive/SciNER/model_v0'

loss_mean = 0
for epoch in range(num_train_epochs):
    # Training
    loss_mean = 0
    model.train()
    for batch in train_dataloader:
        # ['result_float', 'input_ids', 'token_type_ids', 'attention_mask', 'length', 'operation_labels', 'labels']
        model_input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        model_input = {k: v for k,v in batch.items() if k in model_input_keys} # Filter out keys
        outputs = model(**model_input,
                        output_hidden_states=True)

        # print(outputs)

        operand_classification_loss = outputs.loss # Token classification loss 
        # print(operand_classification_loss)

        last_hidden_state = outputs.hidden_states[-1] # Shape: (bs, max_seq_len, hidden_size)
        # print(last_hidden_state.shape)

        lengths = batch['length'] # The total length (before padding)
        # print(lengths)
        op_token_idx = lengths - 2 # Lengths-1 is </s> or [SEP] and before that is [OP]
        # print(op_token_idx)

        op_token_reps = []
        # op_input_ids = [] #For sanity check 

        for i in range(len(op_token_idx)):
            op_token_rep = last_hidden_state[i, op_token_idx[i], :] # Get the op token rep for ith in the batch
            op_token_reps.append(op_token_rep)
            # op_input_ids.append(batch['input_ids'][i, op_token_idx[i]]) #For sanity check 
    
        op_token_reps = torch.stack(op_token_reps).squeeze() # (bs, hidden_size)
        # op_input_ids = torch.stack(op_input_ids).squeeze() #For sanity check 

        # print(batch['input_ids'])
        # print(op_input_ids) #For sanity check 
        # print(op_token_reps.shape)

        # Intent of loop above is to do what's given below, but torch doesn't index as per the tokens like numpy (it gives 8 per example)
        # op_input_ids = batch['input_ids'][:, op_token_idx] # bs, seq_len -> bs 
        # op_token_reps = last_hidden_state[:, op_token_idx, :] # Shape: (bs, hidden_size)

        operation_preds = operation_head(op_token_reps)
        # print(operation_preds.shape, operation_preds)

        operation_classification_loss = F.cross_entropy(operation_preds, batch['operation_labels'])
        # print(operation_classification_loss)


        # loss = weighted_loss(
        #       outputs["logits"].permute(0, 2, 1),
        #       batch["labels"]
        #       )

        total_loss = operand_classification_loss + operation_classification_loss 
        # total_loss = total_loss.item()
        # print(total_loss)
        total_loss.backward()
        # accelerator.backward(total_loss)

        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Observe training loss
    print("Step Loss: {}".format(loss_mean/train_dataloader.__len__()))

    # Evaluation
    model.eval()
    for batch in val_dataloader:
        with torch.no_grad():
            model_input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
            model_input = {k: v for k,v in batch.items() if k in model_input_keys} # Filter out keys
            outputs = model(**model_input, output_hidden_states=True)
            # outputs = model(**batch)

        # Token/OPerand predictions ===== 
        token_predictions = outputs.logits.argmax(dim=-1)
        token_labels = batch["labels"]
        # ==========

        # =========== Computing operation classification preds
        last_hidden_state = outputs.hidden_states[-1] # Shape: (bs, max_seq_len, hidden_size)
        lengths = batch['length'] # The total length (before padding)
        op_token_idx = lengths - 2 # Lengths-1 is </s> or [SEP] and before that is [OP]
        op_token_reps = []
        # op_input_ids = [] #For sanity check 

        for i in range(len(op_token_idx)):
            op_token_rep = last_hidden_state[i, op_token_idx[i], :] # Get the op token rep for ith in the batch
            op_token_reps.append(op_token_rep)
            # op_input_ids.append(batch['input_ids'][i, op_token_idx[i]]) #For sanity check 
    
        op_token_reps = torch.stack(op_token_reps).squeeze() # (bs, hidden_size)
        # op_input_ids = torch.stack(op_input_ids).squeeze() #For sanity check 
        operation_preds = operation_head(op_token_reps).argmax(dim=-1)
        operation_labels = batch['operation_labels']
        # =============================

        # # Necessary to pad predictions and labels for being gathered
        # predictions = accelerator.pad_across_processes(token_predictions, dim=1, pad_index=-100)
        # labels = accelerator.pad_across_processes(token_labels, dim=1, pad_index=-100)

        # predictions_gathered = accelerator.gather(token_predictions)
        # labels_gathered = accelerator.gather(token_labels)

        # true_predictions, true_labels = process_predictions(predictions_gathered, labels_gathered)
        true_predictions, true_labels = process_predictions(token_predictions, token_labels)
        token_metric.add_batch(predictions=true_predictions, references=true_labels)

        # ===== 
        operation_preds = operation_preds.detach().cpu().clone().numpy()
        operation_labels = operation_labels.detach().cpu().clone().numpy()
        op_preds_names = [operation_id2label[p] for p in operation_preds]
        op_labels_names = [operation_id2label[l] for l in operation_labels]
        operation_metric.add_batch(predictions=op_preds_names, references=op_labels_names)

    token_results = token_metric.compute()
    operation_results = operation_metric.compute()
    print(
        f"epoch {epoch}: Token",
        {
            key: token_results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )
    print(
        f"epoch {epoch}: Seq/OPeration",
        {
            key: operation_results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Save and upload
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    model.save_pretrained(f'model_{epoch}_{datetime.datetime.now()}')

    # if accelerator.is_main_process:
    tokenizer.save_pretrained(f'model_{epoch}_{datetime.datetime.now()}')
