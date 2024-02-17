from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd 
import numpy as np 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mawps')

dataset_map = {
    'mawps': 'MU-NLPC/Calc-mawps',
    'svamp': "MU-NLPC/Calc-svamp",
    'asdiv': "MU-NLPC/Calc-asdiv_a"
}

args = parser.parse_args()

dataset = load_dataset(dataset_map[args.dataset])

ops = set(['+', '-', '*', '/'])

def process_row(row):
    col_name = 'expression' if args.dataset == 'mawps' else 'equation'

    exp = row[col_name]

    op_count = {}
    for op in ops:
        if op in exp:
            op_count[op] = exp.count(op)

    eq_count = exp.count('=')
    # print(row['chain'], exp, eq_count)

    row['num_unique_ops'] = len(op_count)
    if len(op_count) != 1 or eq_count != 0:
        row['valid'] = False 
        row['operation'] = np.nan
        return row     
    
    # Only if one unique op present
    row['valid'] = True
    row['operation'] = list(op_count.keys())[0] # First one

    op = row['operation']
    num_occurances = op_count[op] # Number of occurances
    num_operands = num_occurances + 1 # Assuming binary operands 

    # Remove parantheses 
    exp = exp.replace('(', '')
    exp = exp.replace(')', '')
    # exp = exp.replace(' ', '')

    token_word_map = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
    }

    splits = exp.split(op) # Split it on op
    cleaned_op_str = []
    if len(splits) == num_operands: # Once split if all clean
        row['operand'] = []
        for operand in splits:
            row['operand'].append(float(operand))

            cleaned_op_str.append(operand.lstrip().strip()) # Add either int/decimal

            split_on_dot = operand.split('.')

            if len(split_on_dot) == 2 and split_on_dot[-1].lstrip().strip() == '0': # decimal rep and .0
                cleaned_op_str.append(split_on_dot[0].lstrip().strip())

            if 0 <= int(float(operand)) <= 9:
                cleaned_op_str.append(token_word_map[int(float(operand))])

    else:
        pass # TODO handle this

    tag_seq = [1 if token.lower() in cleaned_op_str else 0 for token in row['question'].split(' ')]
    
    if sum(tag_seq) < 2:
        row['valid'] = False 
        row['operation'] = np.nan
        return row     

    question_split = [token for token in row['question'].split(' ')]

    assert len(splits) == num_operands

    row['operand_tags'] = tag_seq 
    row['question_split'] = question_split

    # TODO Validate that the result matches. Call sympy?
    # Validate by recon structing the expression Reconstructed expr 
    reconst_expr = row['operation'].join(splits)
    assert reconst_expr == exp
    return row


def process_data(df: pd.DataFrame, name = ''):
    proc_df = df.apply(process_row, axis=1)
    proc_df_filter = proc_df[proc_df['valid'] == True]    
    print(f'For split {name}, lost {(len(proc_df_filter)/len(df)):2f}% ({len(df)} -> {len(proc_df_filter)})')
    return proc_df_filter
    
   
# For asdiv
def create_equation(row):
    chain = row['chain']

    exp = chain
    # exp = "".join(chain.split()) # Stripping off all whotespace

    # Strip of final result 
    exp = exp.replace('\n', '')
    exp = exp.split('<output>')[0] # Everything before result

    exp = exp.replace('<gadget id="calculator">', '( ')
    exp = exp.replace('</gadget>', ' )')
    # exp = exp.replace('<output>', ' = ')
    # exp = exp.replace('</output>', '')

    row['equation'] = exp

    return row
    

combined_dataset = DatasetDict()

for split in dataset.keys():
    pd_split = pd.DataFrame(dataset[split])

    if args.dataset == 'asdiv':
        pd_split = pd_split.apply(create_equation, axis=1)

    proc_split = process_data(pd_split, split)
    proc_dataset = Dataset.from_pandas(proc_split, split=split)
    combined_dataset[split] = proc_dataset


# pd_train = pd.DataFrame(dataset['train'])
# pd_val = pd.DataFrame(dataset['validation'])
# pd_test = pd.DataFrame(dataset['test'])

# proc_val = process_data(pd_val, 'validation')
# proc_test = process_data(pd_test, 'test')
# proc_train = process_data(pd_train, 'train')


# Assuming you have three pandas dataframes: pd_train, pd_val, pd_test
# train_dataset = Dataset.from_pandas(proc_train, split='train')
# val_dataset = Dataset.from_pandas(proc_val, split='validation')
# test_dataset = Dataset.from_pandas(proc_test, split='test')

# combined_dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})

combined_dataset.push_to_hub(f"vishruthnath/Calc-{args.dataset}-Tagged")