#!/bin/bash

OUTDIR="../out/"
for BERT in bert-base-multilingual-cased # xlm-roberta-base sentence-transformers # labse
do
  TRAINCOR=wikimatrix.tok.hyphen.filtered.target50k_all.and.${BERT}.npz
  MODELDIR=../out/lightning_logs_XLDistillation/${BERT}_${TRAINCOR}/
  for VER in version_2 version_3
  do
    for TASK in MCL-WiC-en-ar MCL-WiC-en-fr MCL-WiC-en-ru MCL-WiC-en-zh
    do
      python ./eval_word-in-context_disentangle.py --target ${TASK} --out ${OUTDIR} --bert ${BERT} --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 --model_path ${MODELDIR}${VER}/
    done
    for TASK in STS17-en-ar STS17-en-de STS17-en-tr STS17-es-en STS17-fr-en STS17-it-en STS17-nl-en
    do
      python ./eval_sts_bertscore.py --target ${TASK} --out ${OUTDIR} --bert ${BERT} --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 --model_path ${MODELDIR}${VER}/
    done
  done
done



