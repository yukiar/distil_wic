import codecs, sys

sentences = []
out_path = sys.argv[2]
with codecs.open(sys.argv[1], 'r', 'utf-8') as f:
    for s_line in f:
        s_line_hyp = s_line.replace(' - ', '-')  # Remove spacing around hyphens
        sentences.append(s_line_hyp)
        if len(sentences) == 100000:
            with codecs.open(out_path, 'a', 'utf-8') as fw:
                fw.writelines(sentences)
            sentences = []

if len(sentences) > 0:
    with codecs.open(out_path, 'a', 'utf-8') as fw:
        fw.writelines(sentences)
