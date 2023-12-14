# Learning to Use the Calculator Improves Numerical Abilities of Language Models
Source code for the project titled as above for CMU's 11-711 ANLP course project
Team Number 26 

You can find our datasets and models at https://huggingface.co/collections/CMU-ANLP-Team-26/

## Encoder- Decoder Based approach

```cd Encoder-Decoder```

For continued pretraining on reframed MAWPS and Multi-NLI data, download the data from https://huggingface.co/datasets/vishwa27/MAWPS-MNLI-CalX-NumEval and place them under the data folder as

```data/train_preft_flant5.csv```
```data/test_preft_flant5.csv```

Change the appropraite ```repo_name``` at line 165 and run 

```python train_flanT5.py```

To evaluate the trained models on specific tasks, firstly download the datasets from AWPNLI, NewsNLI and RTE-Quant data from https://huggingface.co/datasets/NLPFin/Quantitative101

Then run 

```python eval_ft_flanT5.py``` to test FlanT5 (TB-PT ours)

and 

```python eval_fs_flanT5.py``` to test FlanT5 with few-shot prompting

## Encoder-Only Approach 
```cd Encoder-Only```

```data_preprocessing.py``` preprocesses the Calc-MAWPS data [https://huggingface.co/datasets/MU-NLPC/Calc-mawps](https://huggingface.co/datasets/MU-NLPC/Calc-mawps) with annotations required for Calc-BERT continued pretraining. Preprocessed data pushed to [https://huggingface.co/vishruthnath/Calc_BERT_20](https://huggingface.co/vishruthnath/Calc_BERT_20). 

```train.py``` contains the script for continued pretraining with the dual objective. The data used is Calc_BERT_20 linked above from Huggingface. 

```qnli_finetuine.py``` is the script for finetuning Calc-BERT on downstream QNLI tasks. 
