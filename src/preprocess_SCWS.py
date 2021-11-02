import codecs, re, argparse

parser = argparse.ArgumentParser(description='Preprocess CoSimLex.')
parser.add_argument('--input', help='path to the input file', type=str, required=True)
parser.add_argument('--output', help='path to the output file', type=str, required=True)
args = parser.parse_args()

def find_word_in_context(pattern, context):
    result = re.findall(pattern, context)
    if len(result)==1:
        word=result[0][1]
        rep_context = re.sub(pattern, r'\2', context)
    else:
        raise Exception('Match error!')

    return word, rep_context

pattern = re.compile(r'(<b>\s*)(\S+?)(\s*</b>)')

processed = []
with codecs.open(args.input, 'r', encoding='utf-8') as f:
    for line in f:
        a = [x.strip() for x in line.split('\t')]
        w1, c1=find_word_in_context(pattern, a[5])
        w2, c2=find_word_in_context(pattern, a[6])

        # word 1
        a[1] = w1
        # word 2
        a[3] = w2

        # context 1
        a[5] = c1
        # context 2
        a[6] = c2

        processed.append(a)

with codecs.open(args.output, 'w', encoding='utf-8') as fw:
    for a in processed:
        fw.write('\t'.join([entry for entry in a]) + '\n')
