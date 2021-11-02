import codecs, re, argparse

parser = argparse.ArgumentParser(description='Preprocess CoSimLex.')
parser.add_argument('--input', help='path to the input file', type=str, required=True)
parser.add_argument('--output', help='path to the output file', type=str, required=True)
args = parser.parse_args()

def find_word_in_context(pattern, context):
    result = re.findall(pattern, context)
    if len(result) == 2:
        w1 = result[0][1]
        w2 = result[1][1]
        rep_context = re.sub(pattern, r' \2 ', context)
        rep_context = rep_context.replace('  ', ' ')
    else:
        raise Exception('Match error!')

    return w1, w2, rep_context

pattern = re.compile(r'(<strong>\s*)(\S+?)(\s*</strong>)')
processed = []
with codecs.open(args.input, 'r', encoding='utf-8') as f:
    f.readline()  # skip header
    for line in f:
        a = [x.strip() for x in line.split('\t')]
        a_out = [None] * 15
        a_out[0] = a[0]
        a_out[1] = a[1]

        # context 1
        w1, w2, c1 = find_word_in_context(pattern, a[2])
        if (w1 == a[9] and w2 == a[10]) or (w1 == a[10] and w2 == a[9]):
            pass
        else:
            raise Exception('Matching error!')
        a_out[2] = c1
        a_out[3] = c1

        # context 2
        w3, w4, c2 = find_word_in_context(pattern, a[3])
        if (w3 == a[11] and w4 == a[12]) or (w3 == a[12] and w4 == a[11]):
            pass
        else:
            raise Exception('Matching error!')
        a_out[4] = c2
        a_out[5] = c2

        for i in range(4, 13):
            a_out[i + 2] = a[i]

        processed.append(a_out)

with codecs.open(args.output, 'w', encoding='utf-8') as fw:
    for a in processed:
        fw.write('\t'.join([entry for entry in a]) + '\n')
