from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd 
import numpy as np 

dataset = load_dataset("MU-NLPC/Calc-mawps")

ops = set(['+', '-', '*', '/'])
def process_row(row):
    exp = row['expression']
    op_count = {}
    for op in ops:
        if op in exp:
            op_count[op] = exp.count(op)

    row['num_unique_ops'] = len(op_count)
    if len(op_count) != 1:
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

    splits = exp.split(op) # Split it on op
    if len(splits) == num_operands: # Once split if all clean
        row['operands'] = [float(operand) for operand in splits] 
    else:
        pass # TODO handle this


    # TODO get the tag sequence
    tag_seq = [1 if token in splits else 0 for token in row['question'].split(' ')]
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
    

pd_train = pd.DataFrame(dataset['train'])
pd_val = pd.DataFrame(dataset['validation'])
pd_test = pd.DataFrame(dataset['test'])

proc_val = process_data(pd_val, 'validation')
proc_test = process_data(pd_test, 'test')
proc_train = process_data(pd_train, 'train')


# Assuming you have three pandas dataframes: pd_train, pd_val, pd_test
train_dataset = Dataset.from_pandas(proc_train, split='train')
val_dataset = Dataset.from_pandas(proc_val, split='validation')
test_dataset = Dataset.from_pandas(proc_test, split='test')

combined_dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})

combined_dataset.push_to_hub("vishruthnath/Calc-MAWPS-CalcBERT-Tags")