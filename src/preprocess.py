import sys, os, argparse
import stanza
from langdetect import detect
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Preprocess a corpus.')
parser.add_argument('--input', help='path to the corpus', type=str, required=True)
parser.add_argument('--line_num', help='Number of lines in the corpus', type=int, required=True)
args = parser.parse_args()


in_file_name = os.path.basename(args.input)
stanza.download('en')
para_num = 10
stanzas = []
for i in range(para_num):
    stanzas.append(stanza.Pipeline(lang='en', processors='tokenize', use_gpu=True))

buf_size = 1000
bufs = {}
for i in range(para_num):
    bufs[i] = []


def process(proc_id):
    output = []
    doc = stanzas[proc_id](' '.join(bufs[proc_id]))
    for sentence in doc.sentences:
        if len(sentence.tokens) > 4 and len(sentence.tokens) < 51:
            output.append(' '.join([token.text for token in sentence.tokens]))

    with open('../out/' + in_file_name + '_' + str(proc_id) + '.tok', mode='a') as fw:
        fw.write('\n'.join(output) + '\n')


# punc = ['.', '!', '?']
sent_cnt = 0
to_process = False
with open(args.input) as f:
    for line in f:
        s_line = line.strip()
        # if s_line[0].isalpha() and s_line[len(s_line) - 1] in punc: # Removed these by linux command as preprocessing
        if detect(s_line) == 'en':  # Language identification
            bufs[sent_cnt % para_num].append(s_line)
            sent_cnt += 1
            if sent_cnt % (buf_size * para_num) == 0:
                to_process = True

        if to_process:
            result = Parallel(n_jobs=para_num, verbose=10)(
                [delayed(process)(proc_id) for proc_id in range(para_num)])

            for i in range(para_num):
                bufs[i] = []
            per = sent_cnt / args.line_num * 100
            print('processed {0} sentences: {1:.4f} %'.format(sent_cnt, per))

            to_process = False

if len(bufs[0]) > 0:
    result = Parallel(n_jobs=para_num, verbose=10)(
        [delayed(process)(proc_id) for proc_id in range(para_num)])
    per = sent_cnt / args.line_num * 100
    print('processed {0} sentences: {1:.4f} %'.format(sent_cnt, per))
