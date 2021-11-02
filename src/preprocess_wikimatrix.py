from tqdm import tqdm
import stanza, sys
import langdetect, argparse

parser = argparse.ArgumentParser(description='Preprocess a corpus.')
parser.add_argument('--dir_path', help='Path to the directory to wikimatrix', type=str, required=True)
parser.add_argument('--lang_pair', help='Language pair', type=str, required=True)
args = parser.parse_args()

def load_parallel_corpus(file_path):
    src, tgt, score = [], [], []
    with open(file_path) as f:
        for line in f:
            array = line.strip().split('\t')
            if len(array[1]) > 45 and len(array[2]) > 45:
                score.append(float(array[0]))
                src.append(array[1])
                tgt.append(array[2])

    # max_pair_num = int(min(MAX_PAIR, MAX_RATIO * len(src)))
    # return src[:max_pair_num], tgt[:max_pair_num], score[:max_pair_num]
    return src, tgt, score


def tokenize(nlp, sent):
    tokens = []
    doc = nlp(sent)
    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens.append(token.text)
    return tokens


def len_filter(tokens):
    if len(tokens) < 15 and len(tokens) > 51:
        return False
    return True


def lang_filter(langcode, sent):
    try:
        detected_lang = langdetect.detect(sent)
        if langcode == 'zh':
            return detected_lang == 'zh-cn'
        else:
            return detected_lang == langcode
    except langdetect.lang_detect_exception.LangDetectException:
        return False


def to_string_recover_hyphen(toks):
    sent = ' '.join(toks)
    sent = sent.replace(' - ', '-')  # Remove spacing around hyphens
    return sent


def preprocess(slang, tlang, srcs, tgts, scores):
    stanza.download(slang)
    stanza.download(tlang)
    s_stanza = stanza.Pipeline(lang=slang, processors='tokenize', tokenize_no_ssplit=True)
    t_stanza = stanza.Pipeline(lang=tlang, processors='tokenize', tokenize_no_ssplit=True)

    f_srces, f_tgts, f_scores = [], [], []
    for i in tqdm(range(len(srcs))):
        if lang_filter(slang, srcs[i]) and lang_filter(tlang, tgts[i]):
            s_toks = tokenize(s_stanza, srcs[i])
            t_toks = tokenize(t_stanza, tgts[i])
            if len_filter(s_toks) and len_filter(t_toks):
                f_srces.append(s_toks)
                f_tgts.append(t_toks)
                f_scores.append(scores[i])

    return f_srces, f_tgts, f_scores

slang, tlang = tuple(args.lang_pair.split('-'))
srcs, tgts, scores = load_parallel_corpus(args.dir_path + args.lang_pair + '.tsv')
srcs, tgts, scores = preprocess(slang, tlang, srcs, tgts, scores)

if slang == 'en':
    with open(args.dir_path + args.lang_pair + '.tok.hyphen.filtered.tsv', 'w') as fw:
        for i in range(len(srcs)):
            fw.write('{0}\t{1}\t{2}\n'.format(scores[i], to_string_recover_hyphen(srcs[i]),
                                              to_string_recover_hyphen(tgts[i])))
else:  # unify format as en-X
    with open(args.dir_path + 'en-' + slang + '.tok.hyphen.filtered.tsv', 'w') as fw:
        for i in range(len(srcs)):
            fw.write('{0}\t{1}\t{2}\n'.format(scores[i], to_string_recover_hyphen(tgts[i]),
                                              to_string_recover_hyphen(srcs[i])))
