from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

def preprocess_qqa(examples, tokenizer, version='ORG'):
  # version: is either ['original', 'scientific', 'digit']
    version_field_mapping = {
        'ORG': 'question',
        'DIG': 'question_char', 
        'SCI': 'question_sci_10E'
    }
    option_names = ["Option1", "Option2"]

    label_map = {
        'Option 1': 0,
        'Option 2': 1
    }

    question_field = version_field_mapping[version]
    num_options = len(option_names)
    # print('Hi', len(examples), type(examples))

    questions = [[context] * num_options for context in examples[question_field]] # Batch size number of lists, each with 4 copies (one for each option)
    # print(len(questions), questions)

    options = [
        [examples[option][i] for option in option_names] for i in range(len(questions))
    ] 
    # print(len(options), options)

    questions = sum(questions, [])
    options = sum(options, [])

    # print(len(questions), questions) # Len = Num options * Batch size
    # print(len(options), options) # Len = Num Options * Batch size

    tokenized_examples = tokenizer(questions, options, truncation=True) # Handles pairs
    res = {k: [v[i : i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()} # Group each up together

    res['label'] = [label_map[label_name] for label_name in examples['answer']]    
    return res


@dataclass
class DataCollatorForQQA:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None



    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels" # Swag has label integer
        labels = [feature.pop(label_name) for feature in features] # REmove the labels out

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"]) # Num of options
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ] # Flatten all choices
        flattened_features = sum(flattened_features, [])

        # Flatten and pad
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Unflatten them
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch