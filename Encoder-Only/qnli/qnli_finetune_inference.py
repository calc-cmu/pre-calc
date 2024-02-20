
# Load the corresponding dataset from https://huggingface.co/datasets/NLPFin/Quantitative101
import json
from torch.utils.data import Dataset, DataLoader,Subset
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import notebook_login
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import KFold
from transformers import DataCollatorWithPadding

# Loading dataset
# For NewsNLI, AWPNLI, RedditNLI and RTE-Quant
file_path = '/content/NewsNLI.txt'

with open(file_path, 'r') as f_in:
    if file_path.endswith('.txt'):
        lines = f_in.readlines()
        obj =json.loads(lines[0])
    elif file_path.endswith('.json'):
        obj = json.load(f_in)

# For Stress Test
with open("/content/QNLI-Stress_Test_train.txt",'r') as f_in:
    lines = f_in.readlines()
    obj_train =json.loads(lines[0])

with open("/content/QNLI-Stress_Test_test.txt",'r') as f_in:
    lines = f_in.readlines()
    obj_test =json.loads(lines[0])

dataset_train = NLIDataset(obj_train,tokenizer,"sci")
print(len(dataset_train))

dataset_test = NLIDataset(obj_test,tokenizer,"sci")
print(len(dataset_test))

# Tokenization
notebook_login()
# model_name = "vishwa27/CN_BERT_Digit"
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def label_id_map(objects):
    labels = []
    label_to_id = {}
    id_to_label = {}

    for o in objects:
        labels.append(o['answer'])

    uniq_labels = np.unique(labels)

    for i in range(len(uniq_labels)):
      id_to_label[i] = uniq_labels[i]

    for i in range(len(uniq_labels)):
      label_to_id[uniq_labels[i]] = i

    return id_to_label, label_to_id


class NLIDataset(Dataset):

    def __init__(self, objects,tokeizer,datatype):
        sent1 = []
        sent2 = []
        labels = []

        # choosing input feature depending on datatype
        if datatype == "org":
          for o in objects:
            sent1.append(o['statement1'])
            sent2.append(o['statement2'])

            labels.append(o['answer'])

          tokenized_inputs = tokenizer(sent1,sent2,padding=True,truncation=True, max_length=512)

        elif datatype == "digit":
          for o in objects:
            sent1.append(o['statement1_char'])
            sent2.append(o['statement2_char'])

            labels.append(o['answer'])

          tokenized_inputs = tokenizer(sent1,sent2,padding=True,truncation=True, max_length=512)

        else: # sci notation
          for o in objects:
            sent1.append(o['statement1_sci_10E'])
            sent2.append(o['statement2_sci_10E'])
            labels.append(o['answer'])

          tokenized_inputs = tokenizer(sent1,sent2,padding=True,truncation=True, max_length=512)


        print(tokenized_inputs.keys())
        tokenized_inputs['label'] = labels
        self.data = tokenized_inputs
        _, self.label_to_id = label_id_map(objects)

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, i):

        sample = {'input_ids':self.data['input_ids'][i],
                  'token_type_ids':self.data['token_type_ids'][i],
                  'attention_mask':self.data['attention_mask'][i],
                  'labels':self.label_to_id[self.data['label'][i]]}

        return sample

id_to_label, label_to_id = label_id_map(obj_train)

# Change the reframing data to "sci","org" or "digit"
dataset = NLIDataset(obj,tokenizer,"sci")
id_to_label, label_to_id = label_id_map(obj)

# Modeling
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(id_to_label.keys()), id2label=id_to_label, label2id=label_to_id, ignore_mismatched_sizes=True
)

accuracy = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
F1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # print(predictions)
    # print(labels)

    # results = seqeval.compute(predictions=predictions, references=true_labels)
    return {
        "f1": F1_metric.compute(predictions=predictions, references=labels,average='weighted'),
        "accuracy": accuracy.compute(predictions=predictions, references=labels)
    }
    # return accuracy.compute(predictions=predictions, references=labels)


kf = KFold(n_splits=10,random_state=200,shuffle=True) # Except StressTest, we do not have test split so we perform cross-validation
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="vishwa27/BERT_NewsNLI",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    push_to_hub=False,
)
training_args.set_optimizer(name="adamw_torch", learning_rate=1e-5)

# NewsNLI
from torch import nn

for i, (train_index, test_index) in enumerate(kf.split(dataset)):

    train_data = Subset(dataset, train_index)
    val_data = Subset(dataset, test_index)

    # reinitialize the data
    model = AutoModelForSequenceClassification.from_pretrained(
          model_name, num_labels=len(id_to_label.keys()), id2label=id_to_label, label2id=label_to_id
    )
    
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset= train_data,
      eval_dataset = val_data,
      tokenizer=tokenizer,
      # data_collator=data_collator,
      compute_metrics=compute_metrics,
    )

    trainer.train()
trainer.push_to_hub()

## Training for Stress Test
# trainer = Trainer(
#       model=model,
#       args=training_args,
#       train_dataset= dataset_train,
#       eval_dataset = dataset_test,
#       tokenizer=tokenizer,
#       data_collator=data_collator,
#       compute_metrics=compute_metrics,
# )

# trainer.train()
# trainer.push_to_hub()
