#!/bin/bash

python preprocess_wikimatrix.py --dir_path path_to_directory --lang_pair lang_pair_to_process
python extract_freq_words_parallel.py --input path_to_corpus --lang_pair lang_pair_to_process --out_path path_to_output
python postprocess_corpus_parallel.py --corpus_path path_to_corpus --fasttext path_to_magnitude_fastText_models --lang_pair lang_pair_to_process --positive --negative
python merge_multilingual_corpora.py --out_dir path_to_output_directory --corpus_name corpus_name_prefix

# Recommend to precompute representations of all targets, which greatly saves the training time
# Caution: precomputed representations takes a huge space: 10-100 GB depending on the corpus size and pre-trained model
python extract_embedding.py --data ath_to_corpus --bert sentence-transformers --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 --batch 1024 --maxlen 200 --bottom 7 --top 13 --out ../data/
