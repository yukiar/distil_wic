#!/bin/bash

python preprocess.py --input path_to_corpus --line_num number_of_lines_in_corpus
python recover_hyphen.py path_to_corpus path_to_output # Remove spacing around hyphens to for consistency to BERT tokenizer
python extract_freq_words.py --input path_to_corpus --line_num number_of_lines_in_corpus --out_path path_to_output
python postprocess_corpus.py --original path_to_corpus --rtt path_to_round-trip-translation_of_corpus

# Recommend to precompute representations of all targets, which greatly saves the training time
# Caution: precomputed representations takes a huge space: 10-100 GB depending on the corpus size and pre-trained model
python extract_embedding.py --data path_to_corpus --bert bert-large-cased --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --batch 256 --maxlen 200 --bottom 13 --top 25 --out ../data/
