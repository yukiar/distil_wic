#!/bin/bash
## Train monolingual model
python ./distil_WiC_from_pretrainedlm.py --model bert-large-cased --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --batch 128 --nhead 8 --ff_k 4 --data path_to_corpus_or_pre-computed_representations --out ../out/ --neg_sense --init_lr 3.0e-5 --val_check_interval 0.1 --warmup 1000 --dev_size 10000 --gpus 0 --find_lr --lr_sample 300

## Train cross-lingual model
python ./distil_WiC_from_pretrainedlm.py --model sentence-transformers --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 --batch 512 --nhead 8 --ff_k 4 --data path_to_corpus_or_pre-computed_representations --out ../out/ --neg_sense --init_lr 1.0e-5 --val_check_interval 0.25 --warmup 1000 --dev_size 57327 --gpus 0 --find_lr --lr_sample 300
