import datetime

from torch.utils.data import Dataset, DataLoader,Subset
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
import argparse
import json
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def is_num(string):
    if string.isnumeric():
        return True
    else:
        return False
 
def is_float(word):
    if '.' in word and word.replace(".", "").isnumeric():
        return True
    else:
        return False
    
def round_float(num):
    # trailing zeros
    splits = num.split('.')
    if splits[1][0] == '0':
        return str(int(float(num)))
    else:
        return str(float(num))
    
def reformat(sent):
    new_words = []
    words = sent.split(' ')
    for w in words:
        if is_float(w):
            new_words.append(round_float(w))
        else:
            new_words.append(w)
    
    return ' '.join(new_words)

def pad_word_ids(id_list,max_len):
    rem_len = max_len - len(id_list)
    rem_ele = [None] * rem_len

    return id_list + rem_ele

class AWPNLIDataset(Dataset):

    def __init__(self, file_path,tokenizer):
        sent1 = []
        sent2 = []
        labels = []

        with open(file_path, 'r') as f_in:
            if file_path.endswith('.txt'):
                lines = f_in.readlines()
                objects =json.loads(lines[0])
            elif file_path.endswith('.json'):
                objects = json.load(f_in)

        for o in objects:
            # need token level split for operand identification
            sent1.append(reformat(o['statement1']).split(' '))
            sent2.append(reformat(o['statement2']).split(' '))

            labels.append(reformat(o['answer']))

        # print(sent1,sent2)
        questions = []
        for i in range(len(sent1)):
            question = sent1[i]
            question.append('[OP]')
            questions.append(question)
        
        self.labels = [nli_label2id[str_label] for str_label in labels]
        self.tokenized_inputs = tokenizer(questions, is_split_into_words=True, add_special_tokens=True, truncation=True, return_length=True)
        self.word_ids = [self.tokenized_inputs[i].word_ids for i in range(len(self.tokenized_inputs["input_ids"]))]
        self.questions = questions
        self.answers = [sent2[i] for i in range(len(sent2))]
        # print(self.word_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):

        
        sample = {'input_ids':self.tokenized_inputs['input_ids'][i],
                  'token_type_ids':self.tokenized_inputs['token_type_ids'][i],
                  'attention_mask':self.tokenized_inputs['attention_mask'][i],
                  'length': self.tokenized_inputs['length'][i],
                  'nli_label':self.labels[i]}

        return sample

def process_predictions(predictions):
    predictions=predictions.detach().cpu().clone().numpy()
    true_predictions = [[token_id2label[p] for p in prediction] for prediction in predictions]

    return true_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='BERT')
args = parser.parse_args() 

if 'BERT' in args.model:
    # model_id_tokenizer = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
elif 'RoBERTa' in args.model:
    # model_id_tokenizer = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

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

nli_id2label = {
    0: 'Entailment',
    1: 'contradiction'
}
nli_label2id = {v: k for k,v in nli_id2label.items()}

# dataset = datasets.load_dataset('vishruthnath/Calc-MAWPS-CalcBERT-Tags')
tokenized_dataset = AWPNLIDataset('data/QNLI_AWPNLI.json',tokenizer)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

test_dataset = tokenized_dataset
test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=8,shuffle=False)

model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        id2label=token_id2label,
        label2id=token_label2id,
        ignore_mismatched_sizes=True
    )
model.resize_token_embeddings(len(tokenizer)) # Resize embedding to include special token [OP]


# =============== Training loop
optimizer = AdamW(model.parameters(), lr=2e-5)

operation_head = torch.nn.Linear(model.config.hidden_size, len(operation_id2label))
if 'BERT' in args.model:
    checkpoint = torch.load('checkpoints/BERT_operation_head_ep20.pth')
elif 'RoBERTa' in args.model:
    checkpoint = torch.load('checkpoints/RoBERTa_operation_head_ep20.pth') 
operation_head.load_state_dict(checkpoint)

token_metric = evaluate.load("seqeval")
operation_metric = evaluate.load("accuracy")

all_operation_preds = []
all_operand_preds = []

model.eval()
for batch in test_dataloader:
        with torch.no_grad():
            model_input_keys = ['input_ids', 'token_type_ids', 'attention_mask']
            model_input = {k: v for k,v in batch.items() if k in model_input_keys} # Filter out keys
            outputs = model(**model_input, output_hidden_states=True)
            # outputs = model(**batch)

            # Token/OPerand predictions ===== 
            token_predictions = outputs.logits.argmax(dim=-1)
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
            # =============================

            operand_predictions = process_predictions(token_predictions)
            operation_preds = operation_preds.detach().cpu().clone().numpy()

            # print(operation_preds)
            # print(Counter(operand_predictions[0]))
            all_operation_preds.extend(operation_preds)
            all_operand_preds.extend(operand_predictions)

assert len(all_operation_preds) == len(tokenized_dataset.questions)

nli_predictions = []
operands_gt_2  = 0
operands_lt_2  = 0

for i in range(len(tokenized_dataset.questions)):
    
    question_words = tokenized_dataset.questions[i]
    answer_words = tokenized_dataset.answers[i]

    sent_len = len(tokenized_dataset.word_ids[i])
    op_preds = all_operand_preds[i][:sent_len]
    word_ids = tokenized_dataset.word_ids[i]
    operation_pred = operation_id2label[all_operation_preds[i]]

    # Finding operands based on the word_ids
    word_wise_label = {}
    for j in range(len(word_ids)):
        if word_ids[j] == None:
            continue
        else:
            if word_ids[j] not in word_wise_label:
                word_wise_label[word_ids[j]] = op_preds[j]
            else:
                if word_wise_label[word_ids[j]] == 'Operand':
                    continue
                else:
                    word_wise_label[word_ids[j]] = op_preds[j]

    
    operands  = []
    for key in word_wise_label:
        if word_wise_label[key] == 'Operand' and (is_num(question_words[key]) or is_float(question_words[key])):
            operands.append(float(question_words[key]))

    answer = None
    for word in answer_words:
        if (is_num(word) or is_float(word)):
            answer = float(word) # taking the first number of the sequence
            break
    
    # Computing correctness
    if len(operands) > 2:
        # we use first 2 operands
        operands = operands[:2]
        operands_gt_2 += 1

    if len(operands) < 2:
        # default to majority vote
        nli_predictions.append(0)
        operands_lt_2 += 1
        continue

    # entailment considered as 0 here
    if operation_pred == '+':
        comp_ans = operands[0] + operands[1]
    elif operation_pred == '*':
        comp_ans = operands[0] * operands[1]
    elif operation_pred == '/':
        comp_ans = (operands[0]/operands[1]) if operands[0] > operands[1] else (operands[1]/operands[0])
    elif operation_pred == '-':
        comp_ans = (operands[0] - operands[1]) if operands[0] > operands[1] else (operands[1] - operands[0])
        
    if comp_ans == answer:
        nli_predictions.append(0)
    else:
        nli_predictions.append(1)
            
print("finished prediction")

assert len(tokenized_dataset.labels) == len(nli_predictions)

print(args.model)
print("F1 score micro: {}".format(f1_score(tokenized_dataset.labels, nli_predictions, average='micro')))
print("Accuracy: {}".format(accuracy_score(tokenized_dataset.labels, nli_predictions)))
