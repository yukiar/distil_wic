import json, stanza, argparse

parser = argparse.ArgumentParser(description='Preprocess CoSimLex.')
parser.add_argument('--input', help='path to the input directory', type=str, required=True)
args = parser.parse_args()


class WiC:
    def __init__(self, w1, w2, c1, c2, label):
        self.w1 = w1
        self.w2 = w2
        self.c1 = c1
        self.c2 = c2
        self.label = label


def tokenize(nlp, sent):
    tokens = []
    doc = nlp(sent)
    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens.append(token.text)
    return ' '.join(tokens)


def tokenize_postprocess(tokenizer, sent, target):
    if ' ' in target:  # MWE
        tcat = target.replace(' ', '')
        sent = sent.replace(target, tcat)
        target = tcat

    sent = sent.replace(target, ' ' + target + ' ')
    sent = sent.replace('  ', ' ')
    ws_split = sent.split(' ')
    tidx = ws_split.index(target)
    tok_sent = tokenize(tokenizer, ' '.join(ws_split[0:tidx])) + ' ' + target + ' ' \
               + tokenize(tokenizer, ' '.join(ws_split[tidx + 1:]))

    return target, tok_sent


def preprocess(lang_pair, dir):
    stanza.download(lang_pair[0:2])
    s_stanza = stanza.Pipeline(lang=lang_pair[0:2], processors='tokenize', tokenize_no_ssplit=True)
    stanza.download(lang_pair[-2:])
    t_stanza = stanza.Pipeline(lang=lang_pair[-2:], processors='tokenize', tokenize_no_ssplit=True)
    # accents = set(['\'', '’', '”'])

    langs = lang_pair.split('-')
    type = 'dev' if langs[0] == langs[1] else 'test'
    with open(dir + type + '.' + lang_pair + '.data') as f:
        data = json.load(f)
    with open(dir + type + '.' + lang_pair + '.gold') as f:
        gold = json.load(f)

    assert len(data) == len(gold)

    gold_dic = {}
    for dic in gold:
        gold_dic[dic['id']] = dic['tag']

    data_dic = {}
    skip_cases = 0
    for idic in data:
        if langs[0] == langs[1]:
            span1 = tuple([int(idic['start1']), int(idic['end1'])])
            span2 = tuple([int(idic['start2']), int(idic['end2'])])
        else:
            span1 = tuple([int(i) for i in idic['ranges1'].split('-')])
            if ',' in idic['ranges2']:
                # Skip split cases
                skip_cases += 1
                continue
            span2 = tuple([int(i) for i in idic['ranges2'].split('-')])
        w1 = idic['sentence1'][span1[0]:span1[1]]
        w2 = idic['sentence2'][span2[0]:span2[1]]
        w1, c1 = tokenize_postprocess(s_stanza, idic['sentence1'], w1)
        w2, c2 = tokenize_postprocess(t_stanza, idic['sentence2'], w2)

        item = WiC(w1, w2, c1, c2, gold_dic[idic['id']])
        data_dic[idic['id']] = item
    print('####### MCL-WiC: {0} #######'.format(lang_pair))
    print('Skipped {0} cases for split targets. Test sets was reduced to {1}'.format(skip_cases, len(data_dic)))
    return data_dic


def savefile(out_path, data_dic):
    with open(out_path, 'w') as fw:
        for id, wic in data_dic.items():
            fw.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(wic.w1, wic.w2, wic.c1, wic.c2, wic.label))


lang_pairs = ['en-fr', 'en-ru', 'en-ar', 'en-zh', 'ar-ar', 'ru-ru', 'fr-fr', 'zh-zh']
for lang_pair in lang_pairs:
    langs = lang_pair.split('-')
    type = 'dev' if langs[0] == langs[1] else 'test'
    dir = args.input + '/MCL-WiC-' + lang_pair + '/' + type + '/'
    data_dic = preprocess(lang_pair, dir)
    out_path = dir + type + '.' + lang_pair + '.processed.tsv'
    savefile(out_path, data_dic)
