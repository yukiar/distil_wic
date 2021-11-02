#!/bin/bash

for CORPUS in bert-large-cased_wikipedia_target50k_1020k_hyphen.tok.and
do
  MODELDIR="../out/lightning_logs_Distillation/${CORPUS}/"

    for VER in version_0 version_1 version_2 version_3
    do
      LARGE="${MODELDIR}${VER}/"
      OUTDIR="../out/${CORPUS}/${VER}/"

      ## WiC
      for TASK in SCWS CoSimLex USim WiC
      do
      python ./eval_word-in-context_disentangle.py --target ${TASK} --bert bert-large-cased --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --model_path ${LARGE} --out ${OUTDIR} --verbose
      done

      ## STS
      for TASK in STS12 STS13 STS14 STS15 STS16
      do
      python ./eval_sts_bertscore.py --target ${TASK} --bert bert-large-cased --layer 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --model_path ${LARGE} --out ${OUTDIR} --verbose
      done
    done
done