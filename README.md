# Distilling Word Meaning in Context from Pre-trained Language Models
Official repository for Findings-EMNLP2021 paper: Distilling Word Meaning in Context from Pre-trained Language Models

By Yuki Arase

## Train distillers
```python
python distil_WiC_from_pretrainedlm.py --model bert-large-cased --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --batch 128 --nhead 8 --ff_k 4 --data path_to_corpus_or_pre-computed_representations --out ../out/ --neg_sense --init_lr 3.0e-5 --val_check_interval 0.1 --warmup 1000 --dev_size 10000 --gpus 0 --find_lr --lr_sample 300
```

## Evaluate trained distillers
The following script evaluates the trained distillers on CosimLex, USim, SCWS tasks.
```
eval_monolingual.sh
```

We preprocessed the evaluation corpora for simply sentence segmentation and tokenization. 
```
preprocess_eval_corpora_monolingual.sh
```

## Create a training corpus
The training corpus creation consists of the following steps. 
1. Preprocessing (sentence splitting and tokenization, filtering)
2. Round-trip translation to generate paraphrases (please train your own NMT models)
3. Word alignment & masked token prediction
4. (Precompute representations of target words, recommended for faster training)
```
create_training_corpora_monolingual.sh
```

For faster computation of fastText models, you may want to use [Magnitude](https://github.com/plasticityai/magnitude).
