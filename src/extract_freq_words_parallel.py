import random
import re, sys, argparse
import numpy as np
from tqdm import tqdm

random.seed(42)

parser = argparse.ArgumentParser(description='Preprocess a corpus.')
parser.add_argument('--corpus_path', help='path to the corpus', type=str, required=True)
parser.add_argument('--lang_pair', help='language pair', type=str, required=True)
parser.add_argument('--out_path', help='path to the corpus', type=str, required=True)
parser.add_argument('--vocab_size', help='vocabulary size', type=int, default=50000)
parser.add_argument('--max_samples_per_target', help='vocabulary size', type=int, default=100)
parser.add_argument('--laser', help='vocabulary size', type=float, default=1.05)
args = parser.parse_args()

def load_corpus(path):
    srcs, tgts, scores = [], [], []
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')
            scores.append(float(array[0]))
            srcs.append(array[1])
            tgts.append(array[2])
    return np.array(scores), srcs, tgts


def length_filter(words):
    if len(words) >= 10 and len(words) <= 100:
        return True
    else:
        return False

def extract_words_single_occurence(words):
    single_words = set()
    for w in words:
        if words.count(w) == 1:
            single_words.add(w)
    return single_words


def filter_corpus(scores, en_corpus, tgt_corpus, laser_thresh):
    filtered_en_corpus, filtered_tgt_corpus = [], []
    en_all_words, en_single_words = [], []
    indices = np.where(scores > laser_thresh)[0]

    for idx in tqdm(indices):
        en = en_corpus[idx]
        tgt = tgt_corpus[idx]
        en = en.strip()
        tgt = tgt.strip()
        enwords = en.split(' ')
        tgtwords = tgt.split(' ')

        if length_filter(enwords) and length_filter(tgtwords):
            filtered_en_corpus.append(en)
            filtered_tgt_corpus.append(tgt)
            enwords_lower = [w.lower() for w in enwords]
            en_all_words.append(enwords_lower)
            en_single_words.append(extract_words_single_occurence(enwords_lower))

    return filtered_en_corpus, en_all_words, en_single_words, filtered_tgt_corpus


def count_words(corpus_all_words):
    dic = {}
    for words in tqdm(corpus_all_words):
        for w in words:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    sdict = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    return sdict


def construct_vocab(wfreq, vocab_size):
    stop_words = set(['a', 'about', 'after', 'against', 'all', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'been',
                      'being', 'can', 'could', 'de', 'did', 'do', 'for', 'had', 'has', 'have', 'in', 'into', 'is', 'it',
                      'its', 'may', 'might', 'no', 'not', 'of', 'off', 'on', 'or', 'so', 'that', 'the', 'there',
                      'these', 'this', 'those', 'to', 'up', 'was', 'were', 'will', 'with', 'would'])
    skip = int(len(wfreq) * 0.0001)
    vocab = []
    i = 0
    for w, c in tqdm(wfreq.items()):
        if i < skip:
            i += 1
            continue
        if w in stop_words:
            continue
        elif len(vocab) == vocab_size:
            break
        if w.isalpha():
            vocab.append(w)

    return vocab


def match(corpus_single_words, vocab, max_samples_per_target):
    match_results = []
    for w in tqdm(vocab):
        sid_list = []
        for sid, ws in enumerate(corpus_single_words):
            if w in ws:
                sid_list.append(sid)
        match_results.append(sid_list)

    match_results_sample = []
    # freq_list = np.array([len(sid_list) for sid_list in match_results])
    # max_samples_per_target = np.percentile(freq_list, max_samples_param).astype(int)
    for sid_list in match_results:
        if len(sid_list) > max_samples_per_target:
            sid_list = random.sample(sid_list, max_samples_per_target)
        match_results_sample.append(sid_list)

    return match_results_sample


def extract_sentences(vocab, match_results, sample_size):
    len_list = [len(sid_list) for sid_list in match_results]
    if sum(len_list) < sample_size:
        return match_results
    else:
        extract_ids = []
        for _ in vocab:
            extract_ids.append([])

        loop_cnt = 0
        entry_cnt = 0
        while entry_cnt < sample_size:
            for wid, sid_list in enumerate(match_results):
                if len(sid_list) > loop_cnt:
                    extract_ids[wid].append(sid_list[loop_cnt])
                    entry_cnt += 1
            loop_cnt += 1
    return extract_ids


if __name__ == '__main__':
    print('######################################################################')
    print('Process: {0}\n'.format(args.lang_pair))
    print('Loading...')
    scores, en_corpus, tgt_corpus = load_corpus(args.corpus_path)
    en_corpus, en_all_words, en_single_words, tgt_corpus = filter_corpus(scores, en_corpus, tgt_corpus, args.laser)
    # sample_size = min(2000000, len(en_corpus))

    print('Done! Counting words..')
    en_wfreq = count_words(en_all_words)

    # Skip top-most frequent words
    print('Done! Selecting vocab...')
    en_vocab = construct_vocab(en_wfreq, args.vocab_size)

    # Matching
    print('Matching...')
    en_match_results = match(en_single_words, en_vocab, args.max_samples_per_target)

    # Save
    print('Done. Save!')

    with open(args.out_path, 'w') as fw:
        for wid, sid_list in enumerate(en_match_results):
            if len(sid_list) > 0:
                for sid in sid_list:
                    fw.write('{0}\t{1}\t{2}\n'.format(en_vocab[wid], en_corpus[sid], tgt_corpus[sid]))

    with open(args.out_path+'.vocab', 'w') as fw:
        for wid, sid_list in enumerate(en_match_results):
            if len(sid_list) > 0:
                fw.write(en_vocab[wid] + '\n')
    print('######################################################################\n')
