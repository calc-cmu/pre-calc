import json
import math
from tqdm import tqdm
import re
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# replace this with the checkpoint you are interested to evaluate
model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,device_map="cuda:0")

###############
# AWP-NLI
###############

with open('data/QNLI_AWPNLI.json','r') as f:
    lines = f.readlines()

# to reformat decimals to match training distribution
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

prompt_template_zs = """Output if sentence 2 is Entailed or Contradicted by sentence 1. Give a short explanation and only output 1-word answer ENTAILMENT or CONTRADICTION. 
Use the examples below to understand.

sentence1: There were 58 geese and 37 ducks in the marsh
sentence2: 16 more geese were in the marsh.

output: CONTRADICTION

sentence1: Sally has 9 orange balloons and 4 blue balloons  and she found 2 more of the orange balloons.
sentence2: Sally has 11 orange balloons now

output: ENTAILMENT

sentence1: {sent1}
sentence2: {sent2}

output: """

input_prompts_zs = []
labels = []

for obj in awpnli_obj:
    sent1 = reformat(obj['statement1'])
    sent2 = reformat(obj['statement2'])

    ip_prompt = prompt_template_zs.format(sent1=sent1,sent2=sent2)
    input_prompts_zs.append(ip_prompt)

    label = 0 if obj['answer'] == 'Entailment' else 1
    labels.append(label)

device = 'cuda:0'
outputs_zs  = []

batch_size = 8
n_batches = int(len(input_prompts)/batch_size)

for b in tqdm(range(n_batches)):
    texts = input_prompts_zs[b*batch_size:(b+1)*batch_size]
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model_zs.generate(**encoding)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs_zs.extend(generated_texts)

## Evaluate each expression 
pred_labels_zs = []

for o in outputs_zs:
    if o == 'ENTAILMENT':
        pred_labels_zs.append(0)
    else:
        pred_labels_zs.append(1)


print("Metrics for Zero-shot AWP-NLI")
print(accuracy_score(labels[:len(pred_labels)], pred_labels_zs))
print(f1_score(labels[:len(pred_labels)], pred_labels_zs, average='weighted'))
precision_recall_fscore_support(labels[:len(pred_labels)], pred_labels_zs, average='weighted')

###############
# News-NLI or RTE-Quant
###############
with open('data/NewsNLI.txt','r') as f_in:
    lines = f_in.readlines() 
    obj = json.loads(lines[0])

text_prompt_nli = """text-nli: Output if sentence 2 Entails statement 1 or is Neutral - 
sentence1: The 16 NATO members and the 14 countries which used to form the rival Warsaw Pact agreed that there would be significantly less equipment permitted in the area of application in Europe than there was under the original treaty 
sentence2: The NATO has 16 members .

output: text> ENTAILMENT

text-nli: Output if sentence 2 Entails statement 1 or is Neutral - 
sentence1: Following the Declaration of the Establishment of the State of Israel , May 14 , 1948 , seven Arab states entered Palestine and engaged Israeli forces 
sentence2: Israeli forces attacked seven Arab states in 1948 .

output: text> NEUTRAL

text-nli: Output if sentence 2 Entails statement 1 or is Neutral - 
sentence1: {sentence1}
sentence2: {sentence2}

output: """

text_prompt_nli = """text-nli: Output if sentence 2 Entails statement 1 or is Neutral - 
sentence1: {sentence1}
sentence2: {sentence2}"""

input_text_prompts = []
labels =[]

for element in obj:
    sentence1 = element['statement1']
    sentence2 = element['statement2']

    input_prompt = text_prompt_nli.format(sentence1=sentence1,sentence2=sentence2)

    input_text_prompts.append(input_prompt)
    if element['answer'] == 'Entailment':
        labels.append(0)
    elif element['answer'] == 'neutral':
        labels.append(1)
    else:
        print('Error: missing label')

device = 'cuda:0'
outputs  = []

batch_size = 8
n_batches = int(len(input_text_prompts)/batch_size)

for b in tqdm(range(n_batches)):
    texts = input_text_prompts[b*batch_size:(b+1)*batch_size]
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model.generate(**encoding)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs.extend(generated_texts)

pred_labels = []

for o in outputs:
    if 'ENTAILMENT' in o:
        pred_labels.append(0)
    elif 'NEUTRAL' in o:
        pred_labels.append(1)
    else:
        print('format error')
        pred_labels.append(0)

print("Metrics for Zero-shot News-NLI")
print(accuracy_score(labels[:len(pred_labels)], pred_labels))
print(f1_score(labels[:len(pred_labels)], pred_labels, average='weighted'))
precision_recall_fscore_support(labels[:len(pred_labels)], pred_labels, average='weighted')