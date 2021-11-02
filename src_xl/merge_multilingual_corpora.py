import os, random, argparse

random.seed(42)

parser = argparse.ArgumentParser(description='Distill WiC representation.')
parser.add_argument('--out_dir', help='path to the output directory', type=str, required=True)
parser.add_argument('--corpus_name', help='output corpus name', type=str, required=True)
args = parser.parse_args()


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


def file_reset(out_file_path):
    if os.path.exists(out_file_path):
        print('remove {0}'.format(out_file_path))
        os.remove(out_file_path)


def add_data(out_file_path, indices, targets, sents):
    with open(out_file_path, 'a') as fw:
        for idx in indices:
            fw.write('{0}\t{1}\n'.format(targets[idx], sents[idx]))


if __name__ == '__main__':
    en_out_path = args.out_dir + args.corpus_name + '.and'
    pos_out_path = args.out_dir + args.corpus_name + '.positive_and'
    neg_out_path = args.out_dir + args.corpus_name + '.negative_sense_and'
    lang_pairs = ['en-zh', 'en-de', 'en-et', 'en-ar', 'en-fr', 'en-ru', 'en-ro', 'en-tr', 'en-es']

    file_reset(en_out_path)
    file_reset(pos_out_path)
    file_reset(neg_out_path)

    for lang_pair in lang_pairs:
        tgt_lang = lang_pair.replace('en-', '')
        if os.path.exists(args.out_dir + lang_pair + '.tok.hyphen.filtered.target50k_all.en.p.tmp'):
            sample_type = 'target50k_all'
        else:
            sample_type = 'target50k_1m'
        en_targets, en_sents, en_indices = load_postprocessed_corpus(
            args.out_dir + lang_pair + '.tok.hyphen.filtered.' + sample_type + '.en.p.tmp')
        pos_targets, pos_sents, pos_indices = load_postprocessed_corpus(
            args.out_dir + lang_pair + '.tok.hyphen.filtered.' + sample_type + '.' + tgt_lang + '.p.tmp')
        neg_targets, neg_sents, neg_indices = load_postprocessed_corpus(
            args.out_dir + lang_pair + '.tok.hyphen.filtered.' + sample_type + '.' + tgt_lang + '.n.tmp')

        common_indices = en_indices & pos_indices & neg_indices
        add_data(en_out_path, common_indices, en_targets, en_sents)
        add_data(pos_out_path, common_indices, pos_targets, pos_sents)
        add_data(neg_out_path, common_indices, neg_targets, neg_sents)
        print('{0}\t{1}'.format(lang_pair, len(common_indices)))

    # Shuffle dataset & re-save
    en_targets, en_sents, en_indices = load_postprocessed_corpus(en_out_path)
    pos_targets, pos_sents, pos_indices = load_postprocessed_corpus(pos_out_path)
    neg_targets, neg_sents, neg_indices = load_postprocessed_corpus(neg_out_path)
    common_indices = list(en_indices & pos_indices & neg_indices)
    random.shuffle(common_indices)

    file_reset(en_out_path)
    file_reset(pos_out_path)
    file_reset(neg_out_path)
    add_data(en_out_path, common_indices, en_targets, en_sents)
    add_data(pos_out_path, common_indices, pos_targets, pos_sents)
    add_data(neg_out_path, common_indices, neg_targets, neg_sents)
