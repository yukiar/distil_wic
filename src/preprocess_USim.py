import xml.etree.ElementTree as ET
import html, argparse
from util import smart_lower

parser = argparse.ArgumentParser(description='Preprocess CoSimLex.')
parser.add_argument('--input_xml', help='path to the input xml file', type=str, required=True)
parser.add_argument('--input_annotation', help='path to the input annotation file', type=str, required=True)
parser.add_argument('--output', help='path to the output file', type=str, required=True)
args = parser.parse_args()

vocab = set()
sentences = {}
tree = ET.parse(args.input_xml)
for child in tree.getroot():  # lexelt
    for instance in child.findall('instance'):  # instance
        id = int(instance.attrib['id'])
        for context in instance.iter('context'):  # context
            txt_parts = [s.strip() for s in context.itertext()]
            if len(txt_parts) == 2 or len(txt_parts) == 3:
                for head in context.iter('head'):  # head
                    target = smart_lower(head.text)
            else:
                raise ValueError('Unexpected format')
        vocab.add(target)
        txt = html.unescape(' '.join(txt_parts))
        sentences[id] = (target, txt)

annotations = {}
with open(args.input_annotation, 'r') as f:
    f.readline()  # skip header
    for line in f:
        a = line.split(',')
        if a[3] == 'avg':  # Use only average score
            annotations[(int(a[0]), int(a[1]))] = float(a[2])

# Output
with open(args.output, 'w') as fw:
    for pair, score in annotations.items():
        w1, s1 = sentences[pair[0]]
        w2, s2 = sentences[pair[1]]
        fw.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(w1, w2, s1, s2, score))
