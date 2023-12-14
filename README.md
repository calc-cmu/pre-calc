## Encoder- Decoder Based approach

For continued pretraining on reframed MAWPS and Multi-NLI data, download the data from https://huggingface.co/datasets/vishwa27/MAWPS-MNLI-CalX-NumEval and place them under the data folder as

```data/train_preft_flant5.csv```
```data/test_preft_flant5.csv```

Change the appropraite ```repo_name``` at line 165 and run 

```python train_flanT5.py```

To evaluate the trained models on specific tasks, firstly download the datasets from AWPNLI, NewsNLI and RTE-Quant data from https://huggingface.co/datasets/NLPFin/Quantitative101

Then run 

```python ft_flanT5_eval.py``` to test FlanT5 (ST ours) or FlanT5 (TB-PT ours)

and 

```python fs_flanT5_eval.py``` to test FlanT5 with few-shot prompting