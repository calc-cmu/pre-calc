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
