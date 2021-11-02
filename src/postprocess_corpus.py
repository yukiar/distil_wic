import sys, random, gensim, tqdm, nltk, unicodedata, re, argparse
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import zscore
from nltk.stem import SnowballStemmer
from masked_token_prediction import BERTMaskPrediction

random.seed(42)

parser = argparse.ArgumentParser(description='Distill WiC representation.')
parser.add_argument('--original', help='path to the original corpus', type=str, required=True)
parser.add_argument('--rtt', help='path to the round-trip translation of the original corpus', type=str, required=True)
args = parser.parse_args()


def load_corpus(path):
    targets, sentences = [], []
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')
            targets.append(array[0])
            sentences.append(array[1])

    return targets, sentences


def surface_similar(target, w, stemmer):
    t_stem = stemmer.stem(target)
    w_stem = stemmer.stem(w)
    edist = nltk.edit_distance(target, w, substitution_cost=2)
    if t_stem == w_stem:
        return True
    elif min(len(target), len(w)) <= 3:
        if edist <= 2:
            return True
    else:
        if edist <= 3:
            return True
    return False


def load_neg_corpus(path):
    targets, sentences, inc_idxs = [], [], []
    idx = 0
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')
            if array[0] == '###':
                pass
            else:
                inc_idxs.append(idx)
                targets.append(array[0])
                sentences.append(array[1])
            idx += 1

    return targets, sentences, set(inc_idxs)


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


def generate_negative_examples():
    ### Parameters to Set ###
    K = 100
    T = 0.001
    pred_num_thresh = 3
    #########################
    maskpredictor = BERTMaskPrediction()
    ftmodel = gensim.models.fasttext.load_facebook_model('../data/wiki.en.bin')

    targets, sents = load_corpus(args.original)

    candidates = []
    for i in tqdm.tqdm(range(len(targets)), desc='[Prediction]'):
        t = targets[i]
        s = sents[i]
        # pred_toks, probs=guess_single(tokenizer, model, 'The capital of France is Paris.', 'France', K)

        pred_toks, probs = maskpredictor.guess_masked_token(ftmodel.wv.vocab, s, t, K, T)

        cand = '###'
        if len(pred_toks) >= pred_num_thresh:
            tvec = ftmodel.wv[t.lower()]
            ft_sims = [cosine(tvec, ftmodel.wv[cand.lower()]) for cand in pred_toks]
            sim_mean = sum(ft_sims) / len(ft_sims)

            for c, sim in zip(pred_toks, ft_sims):
                if sim > sim_mean:
                    cand = c
                    break
        candidates.append(cand)

    all_sentences = []
    success_indices = set()
    for i in tqdm.tqdm(range(len(targets)), desc='[Save]'):
        if candidates[i] == '###':
            all_sentences.append('###')
        else:
            success_indices.add(i)
            all_sentences.append(sents[i].replace(targets[i], candidates[i]))
    print('Failed to generate {0} negative samples out of {1} sentences'.format(len(success_indices), len(sents)))
    return candidates, all_sentences, success_indices


def is_noisy(words):
    for w in words:
        if re.search(r'(.)\1{3,}',
                     w):  # Remove a sentence in which the same character sequencially continues more than or equal to four times: a--------b
            return True
        for c in w:
            if unicodedata.east_asian_width(
                    c) in 'FWH':  # Remove a sentence with Hiragana, hankaku-katakana, and multi-byte chars
                return True
    return False


def compute_word_cdist_matrix(sws, tws, ftmodel):
    m = np.zeros((len(sws), len(tws)))
    for i, sw in enumerate(sws):
        for j, tw in enumerate(tws):
            cosdist = cosine(ftmodel.wv[sw.lower()], ftmodel.wv[tw.lower()])
            m[i, j] = np.round(cosdist, 4)

    return m


def generate_positive_examples():
    targets, sents = load_corpus(args.original)
    dist_max_thresh = 0.4
    K = 100
    T = 0.003
    zscore_thresh = -1.0
    stemmer = SnowballStemmer('english')

    print('loading BERT Maksed Token Predictor...')
    maskpredictor = BERTMaskPrediction()
    print('loading FastText...')
    ftmodel = gensim.models.fasttext.load_facebook_model('../data/wiki.en.bin')
    print('done.')

    # Align a target word to a word in positive sentence
    idx = 0
    exception_cnt = 0
    succsess_rtt_cnt = 0
    succsess_bert_cnt = 0
    accept_index = set()
    all_positive_sents = []
    all_positive_targets = []

    with open(args.original) as f:
        for s in f:
            cand_t, cand_sent = None, None
            trans = s.strip().replace(' @-@ ', '-')
            words = trans.split(' ')
            ori_words = sents[idx].split(' ')
            tidx = ori_words.index(targets[idx])

            if s.count('@-@') > 3 or is_noisy(words):
                exception_cnt += 1
                cand_t = '###'
                cand_sent = trans
            else:
                tvec = ftmodel.wv[targets[idx].lower()]
                align_cands = [w for w in words if surface_similar(targets[idx], w, stemmer)]

                if len(align_cands) > 0:
                    index_diff = np.array([abs(tidx - words.index(c)) for c in align_cands])
                    index_diff_order = np.argsort(index_diff)

                    if tidx > 0 and targets[idx][0].isupper():  # Do not replace possible named entities
                        cand_t = align_cands[index_diff_order[0]]
                        cand_sent = trans
                    else:
                        for i in index_diff_order:
                            # BERT mask prediction
                            pred_toks, probs = maskpredictor.guess_masked_token(ftmodel.wv.vocab, trans, align_cands[i],
                                                                                K,
                                                                                T)
                            ft_sims_cands = np.array([cosine(tvec, ftmodel.wv[cand.lower()]) for cand in pred_toks])
                            for c, sim in zip(pred_toks, ft_sims_cands):
                                if sim < dist_max_thresh and not surface_similar(targets[idx], c, stemmer):
                                    cand_t = c
                                    cand_sent = trans.replace(align_cands[i], c)
                                    succsess_bert_cnt += 1
                                    break
                            if cand_t is not None:
                                break

                        if cand_t is None:  # If no replacement is possible, use the word as is
                            cand_t = align_cands[index_diff_order[0]]
                            cand_sent = trans

                    accept_index.add(idx)
                else:
                    m = compute_word_cdist_matrix(ori_words, words, ftmodel)
                    # Detect 'taken' words
                    a = []
                    for i in list(range(0, tidx)) + list(range(tidx + 1, len(ori_words))):
                        min_t_id = np.argmin(m[i])
                        if i == np.argmin(m[:, min_t_id]):
                            a.append(min_t_id)
                    # Find possible alignment for the target
                    mz = zscore(m, axis=0)
                    for cand_id in np.argsort(mz[tidx]):
                        if cand_id not in a and mz[tidx, cand_id] <= zscore_thresh:
                            cand_t = words[cand_id]
                            cand_sent = trans
                            succsess_rtt_cnt += 1
                            accept_index.add(idx)
                            break

                    if cand_t is None:
                        exception_cnt += 1
                        cand_t = '###'
                        cand_sent = trans

            all_positive_targets.append(cand_t)
            all_positive_sents.append(cand_sent)
            idx += 1
            if len(all_positive_targets) % 10000 == 0:
                print(
                    'Processed sentences: {0}\t Lexical paraphrase succeeded: RTT {1} BERT {2}\t RTT failed: {3}'.format(
                        len(all_positive_sents),
                        succsess_rtt_cnt,
                        succsess_bert_cnt,
                        exception_cnt))

    print('Done!')
    print('All sentences: {0}\t Lexical paraphrase succeeded: RTT {1} BERT {2}\t RTT failed: {3}'.format(
        len(all_positive_sents),
        succsess_rtt_cnt, succsess_bert_cnt,
        exception_cnt))
    return all_positive_targets, all_positive_sents, accept_index


def save_results(out_path, acceptids, targets, sents):
    print('saveing...')
    with open(out_path, 'w') as fw:
        for i in acceptids:
            fw.write('{0}\t{1}\n'.format(targets[i], sents[i]))


if __name__ == '__main__':
    # Post-process the negative-sense corpus and sampling
    all_negative_targets, all_negative_sents, acceptidx_negatives = generate_negative_examples()

    # Post-process positive corpus
    all_positive_targets, all_positive_sents, acceptidx_positives = generate_positive_examples()

    # Merge and Save results
    acceptidxs = acceptidx_negatives & acceptidx_positives
    targets, all_sents = load_corpus(args.original)
    save_results(args.original + '.positive_and', acceptidxs, all_positive_targets, all_positive_sents)
    save_results(args.original + '.negative_sense_and', acceptidxs, all_negative_targets, all_negative_sents)
    save_results(args.original + '.and', acceptidxs, targets, all_sents)
