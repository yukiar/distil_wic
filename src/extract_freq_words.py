import random, re, argparse

parser = argparse.ArgumentParser(description='Preprocess a corpus.')
parser.add_argument('--corpus_path', help='path to the corpus', type=str, required=True)
parser.add_argument('--out_path', help='path to the corpus', type=str, required=True)
parser.add_argument('--vocab_size', help='vocabulary size', type=int, default=50000)
parser.add_argument('--sample_size', help='vocabulary size', type=int, default=1000000)
args = parser.parse_args()

def load_corpus(path):
    corpus = []
    corpus_all_words = []
    corpus_single_words = []
    with open(path) as f:
        for line in f:
            str = line.strip()
            words = str.split(' ')
            if len(words) >= 15 and len(words) <= 50:
                if re.match(r'[A-Z]+', str):  # should start by capital alphabet
                    hyphens = re.findall(r'\-', str)
                    numbers = re.findall(r'\d+[,\.\-]*\d*', str)
                    commas = re.findall(r',', str)
                    if len(hyphens) <= 1 and len(numbers) <= 1 and len(commas) <= 2:
                        corpus.append(str)
                        words = list(str.split(' '))
                        single_words = set()
                        for w in words:
                            if words.count(w) == 1:
                                single_words.add(w)
                        corpus_all_words.append(words)
                        corpus_single_words.append(single_words)

    return corpus, corpus_all_words, corpus_single_words


def count_words(corpus_all_words):
    dic = {}
    for words in corpus_all_words:
        for w in words:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    sdict = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    return sdict


def match(corpus_single_words, vocab):
    mcorpus = {}
    for w in vocab:
        mcorpus[w] = []
    for i, ws in enumerate(corpus_single_words):
        for matchd_w in ws & vocab:
            mcorpus[matchd_w].append(i)
    sorted_mcorpus = dict(sorted(mcorpus.items(), key=lambda item: len(item[1]), reverse=True))
    return sorted_mcorpus


if __name__ == '__main__':
    print('Loading...')
    corpus, corpus_all_words, corpus_single_words = load_corpus(args.corpus_path)
    print('Done! Counting words..')
    wfreq = count_words(corpus_all_words)

    # Skip top-most frequent words
    print('Done! Selecting vocab...')
    skip = int(len(wfreq) * 0.0001)
    vocab = set()
    i = 0
    for w, c in wfreq.items():
        if i < skip:
            i += 1
            continue
        elif len(vocab) == args.vocab_size:
            break
        if w.isalpha():
            vocab.add(w)

    # Matching
    print('Matching...')
    mcorpus = match(corpus_single_words, vocab)

    # Sampling
    print('Done! Sampling...')
    out_corpus = []
    used_sent_idx = set()
    while len(out_corpus) < args.sample_size:
        for w, list in mcorpus.items():
            if len(list) > 0:
                ridx = random.choice(list)
                if ridx not in used_sent_idx:
                    out_corpus.append((w, ridx))
                    if len(out_corpus) == args.sample_size:
                        break
                    used_sent_idx.add(ridx)

    # Save
    print('Done. Save!')
    with open(args.out_path, 'w') as fw:
        for item in out_corpus:
            fw.write('{0}\t{1}\n'.format(item[0], corpus[item[1]]))

    vocab = set()
    for term, _ in out_corpus:
        vocab.add(term)
    with open(args.out_path+'.vocab', 'w') as fw:
        for v in vocab:
            fw.write(v + '\n')
