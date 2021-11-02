import sys, random, argparse
import numpy as np
from collections import OrderedDict

import torch
from scipy.stats import zscore, t
from scipy.spatial.distance import cosine
from tqdm import tqdm
from masked_token_prediction import BERTMaskPrediction
from pymagnitudelight import *

random.seed(42)

parser = argparse.ArgumentParser(description='Distill WiC representation.')
parser.add_argument('--corpus_path', help='path to the corpus', type=str, required=True)
parser.add_argument('--lang_pair', help='language pair', type=str, required=True)
parser.add_argument('--fasttext', help='path to the aligned fasttext directory', type=str, required=True)
parser.add_argument('--positive', help='generate positive samples', action='store_true')
parser.add_argument('--negative', help='generate negative samples', action='store_true')
args = parser.parse_args()


def load_parallel_corpus(filepath):
    targets, ens, tgts = [], [], []
    with open(filepath) as f:
        for line in f:
            array = line.strip().split('\t')
            targets.append(array[0])
            ens.append(array[1])
            tgts.append(array[2])

    return targets, ens, tgts


def load_postprocessed_corpus(path):
    targets, sentences, inc_idxs = [], [], []
    idx = 0
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')

            if array[0] != '###':
                inc_idxs.append(idx)

            targets.append(array[0])
            sentences.append(array[1])
            idx += 1

    return targets, sentences, set(inc_idxs)


def save_corpus(out_path, targets, sents, sample_idxs):
    with open(out_path, 'w') as fw:
        for i in sample_idxs:
            fw.write('{0}\t{1}\n'.format(targets[i], sents[i]))


def generate_negative_sense(en_targets, positive_targets, tgts, succsess_indices, lang_pair):
    ### Parameters to Set ###
    K = 30
    T = 0.001
    pred_num_thresh = 3
    #########################
    print('loading XLM-R...')
    maskpredictor = BERTMaskPrediction(multilingual=True, max_length=200)
    print('done')
    print('loading FastText...')
    en_vecs = Magnitude(args.fasttext + '/wiki.en.align.magnitude')
    tgt_vecs = Magnitude(args.fasttext + '/wiki.' + lang_pair.replace('en-', '') + '.align.magnitude')
    print('done.')

    candidates = []
    replaced_tgts = []
    exception_cnt = 0
    succsess_cnt = 0
    with tqdm(range(len(positive_targets)), ncols=100, unit='sents', desc=lang_pair) as pbar:
        for iter_id in pbar:
            if iter_id in succsess_indices:
                t = positive_targets[iter_id]
                s = tgts[iter_id]
                pred_toks, probs = maskpredictor.guess_masked_token(tgt_vecs, s, t, K, T)

                if len(pred_toks) >= pred_num_thresh:
                    pred_toks_lower = [pred.lower() for pred in pred_toks]
                    # Select candidates less similar to positive_target
                    m = compute_word_cdist_matrix([t.lower()], pred_toks_lower, tgt_vecs, tgt_vecs)
                    dist_mean = sum(m[0]) / len(m[0])

                    # Select candidates less similar to english target than positive_target
                    pos_en_cdist = np.round(
                        cosine(en_vecs.query(en_targets[iter_id].lower()), tgt_vecs.query(t.lower())), 4)
                    m_en = compute_word_cdist_matrix([en_targets[iter_id].lower()], pred_toks_lower,
                                                     en_vecs, tgt_vecs)
                    filter_condition = np.where((m[0] > dist_mean) & (m_en[0] > pos_en_cdist))[0]

                    if len(filter_condition) > 0:
                        cand = pred_toks[filter_condition[0]]
                        replaced_s = s.replace(t, cand)
                        succsess_cnt += 1
                    else:
                        cand = '###'
                        replaced_s = s
                        exception_cnt += 1
                else:
                    cand = '###'
                    replaced_s = s
                    exception_cnt += 1
            else:
                cand = '###'
                replaced_s = tgts[iter_id]
                exception_cnt += 1

            candidates.append(cand)
            replaced_tgts.append(replaced_s)
            pbar.set_postfix(OrderedDict(success=succsess_cnt, fail=exception_cnt))

    print('Done!')
    print(
        'Processed ALL sentences: {0}\t Masked Token Prediction succeeded: {1} \t failed: {2}'.format(
            len(positive_targets),
            succsess_cnt,
            exception_cnt))

    return candidates, replaced_tgts


def compute_word_cdist_matrix(en_words, tgt_words, en_vecs, tgt_vecs):
    m = np.zeros((len(en_words), len(tgt_words)))
    for i, sw in enumerate(en_words):
        evec = en_vecs.query(sw)
        for j, tw in enumerate(tgt_words):
            tvec = tgt_vecs.query(tw)
            cosdist = cosine(evec, tvec)
            m[i, j] = np.round(cosdist, 4)

    return m


def generate_positive_sense(targets, ens, tgts, lang_pair):
    zscore_thresh = -1.282  # About 80% confidence interbal in the normal distribution

    print('loading FastText...')
    en_vecs = Magnitude(args.fasttext + '/wiki.en.align.magnitude')
    tgt_vecs = Magnitude(args.fasttext + '/wiki.' + lang_pair.replace('en-', '') + '.align.magnitude')
    print('done.')

    # Align a target word to a word in positive sentence
    exception_cnt = 0
    succsess_align_cnt = 0
    all_positive_targets = []
    all_targets = []  # For true-casing

    with tqdm(range(len(targets)), ncols=100, unit='sents', desc=lang_pair) as pbar:
        for iter_id in pbar:
            target = targets[iter_id]
            en = ens[iter_id]
            tgt = tgts[iter_id]

            en_words = en.split(' ')
            tgt_words = tgt.split(' ')
            en_words_lower = [w.lower() for w in en_words]  # lowercase
            tgt_words_lower = [w.lower() for w in tgt_words]  # lowercase
            tidx = en_words_lower.index(target)
            all_targets.append(en_words[tidx])

            m = compute_word_cdist_matrix(en_words_lower, tgt_words_lower, en_vecs, tgt_vecs)

            # Detect 'taken' words
            a = list()
            min_cosds = list()
            for i in list(range(0, tidx)) + list(range(tidx + 1, len(en_words_lower))):
                min_t_id = np.argmin(m[i])
                if i == np.argmin(m[:, min_t_id]):
                    a.append(min_t_id)
                    min_cosds.append(np.amin(m[i]))
            # Find possible alignment for the target
            mz = zscore(m, axis=0)
            mean_cosd_alinged = np.mean(min_cosds)
            sd_cosd_aligned = np.std(min_cosds)
            if sd_cosd_aligned == 0:
                interval_mean_cosd_alinged = mean_cosd_alinged
            else:
                interval_mean_cosd_alinged = t.interval(alpha=0.5, df=len(min_cosds) - 1, loc=mean_cosd_alinged,
                                                        scale=sd_cosd_aligned)[1]
            filter = np.where((mz[tidx] <= zscore_thresh) & (m[tidx] <= interval_mean_cosd_alinged))[0]

            filter = np.array([idx for idx in filter if idx not in a])
            if len(filter) > 0:
                best_fidx = np.argsort(m[tidx][filter])[0]
                cand_t = tgt_words[filter[best_fidx]]
                succsess_align_cnt += 1
            else:
                exception_cnt += 1
                cand_t = '###'

            all_positive_targets.append(cand_t)

            pbar.set_postfix(OrderedDict(success=succsess_align_cnt, fail=exception_cnt))

    print('Done!')
    print(
        'Processed ALL sentences: {0}\t Lexical paraphrase succeeded: {1} \t failed: {2}'.format(
            len(all_positive_targets),
            succsess_align_cnt,
            exception_cnt))
    return all_targets, all_positive_targets


def save_results(out_path, targets, sents):
    print('saveing...')
    with open(out_path, 'w') as fw:
        for t, s in zip(targets, sents):
            fw.write('{0}\t{1}\n'.format(t, s))


if __name__ == '__main__':
    ### outputs ###
    en_out_path = args.corpus_path + '.en.p.tmp'
    positive_out_path = args.corpus_path + args.lang_pair.replace('en-', '') + '.p.tmp'
    negative_out_path = args.corpus_path + args.lang_pair.replace('en-', '') + '.n.tmp'
    #########################

    # Load parallel corpus
    targets, ens, tgts = load_parallel_corpus(args.corpus_path)

    if args.positive:
        # Post-process positive corpus
        all_targets, all_positive_targets = generate_positive_sense(targets, ens, tgts, args.lang_pair)
        save_results(positive_out_path, all_positive_targets,
                     tgts)  # Save true-cased aligned candidcates with target sentences
        save_results(en_out_path, all_targets, ens)  # Save true-cased targets with source sentences

    if args.negative:
        # Post-process the negative-sense corpus and sampling
        all_positive_targets, tgts, succsess_indices = load_postprocessed_corpus(positive_out_path)
        all_negative_targets, replaced_tgts = generate_negative_sense(targets, all_positive_targets, tgts,
                                                                      succsess_indices,
                                                                      args.lang_pair)
        save_results(negative_out_path, all_negative_targets,
                     replaced_tgts)  # Save true-cased aligned candidcates with target sentences
