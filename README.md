# Pre-Calc – Learning to Use the Calculator Improves Numeracy in Language Models

This work has been accepted at SemEval NAACL 2024, please find the paper here: https://drive.google.com/file/d/1-XCOW0o27mTz7gNJe48SNfTIU_-0DHzr/view?usp=sharing

Source code for the project titled as above

You can find our datasets and models at https://huggingface.co/collections/Calc-CMU/pre-calc-657a5ad5f1ae42fb12364563

## Encoder-Only Approach 
```cd Encoder-Only```

```data_preprocessing.py``` preprocesses the Calc-MAWPS data [https://huggingface.co/datasets/MU-NLPC/Calc-mawps](https://huggingface.co/datasets/MU-NLPC/Calc-mawps) with annotations required for Calc-BERT continued pretraining. Preprocessed data pushed to [https://huggingface.co/vishruthnath/Calc_BERT_20](https://huggingface.co/vishruthnath/Calc_BERT_20). 

```train.py``` contains the script for continued pretraining with the dual objective. The data used is Calc_BERT_20 linked above from Huggingface. 

```qnli_finetuine_inference.py``` is the script for finetuning Calc-BERT on downstream QNLI tasks. 

```inference_awpnli.py``` for inference of Pre-Calc model on AWPNLI Tasks

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
