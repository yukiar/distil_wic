import argparse, torch
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import SequentialSampler, BatchSampler
from pretrained_model import Pretrained_Model

parser = argparse.ArgumentParser(description='Extract embeddings.')
parser.add_argument('--data', help='dataset', type=str, required=True)
parser.add_argument('--out', help='dataset', type=str, default='../out/')
parser.add_argument('--bert', help='BERT model name', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
                             'xlm-roberta-base','xlm-roberta-large', 'sentence-transformers', 'labse', 'bert-base-multilingual-cased'])
parser.add_argument('--layer', nargs='+', help='layer to use', type=int, default=12)
parser.add_argument('--bottom', help='batch size', type=int, default=7)
parser.add_argument('--top', help='batch size', type=int, default=13)
parser.add_argument('--maxlen', help='dataset', type=int, default=512)
parser.add_argument('--batch', help='Batch size', type=int, default=128)
args = parser.parse_args()


class Embedder:
    def __init__(self, bottom, top, targets, sents, positive_targets, positive_sents, neg_sense_targets,
                 neg_sense_sents):
        # Load pre-trained model
        self.premodel = Pretrained_Model(args.bert, args.layer, max_length=args.maxlen)
        self.targets = targets
        self.sents = sents
        self.positive_targets = positive_targets
        self.positive_sents = positive_sents
        self.neg_sense_targets = neg_sense_targets
        self.neg_sense_sents = neg_sense_sents
        self.bottom = bottom
        self.top = top

    def encode_by_bert(self, indices):
        src_vecs = self.premodel.encode([self.targets[i] for i in indices], [[self.sents[i]] for i in indices],
                                        match_all=False, mean=False)

        tgt_vecs = self.premodel.encode([self.positive_targets[i] for i in indices],
                                        [[self.positive_sents[i]] for i in indices], match_all=False, mean=False)

        neg_sense_vecs = self.premodel.encode([self.neg_sense_targets[i] for i in indices],
                                              [[self.neg_sense_sents[i]] for i in indices], match_all=False,
                                              mean=False)

        # Remove Nan
        src_no_nan_indices = set([i for i in range(len(indices)) if not torch.any(torch.isnan(src_vecs[i]))])
        tgt_no_nan_indices = set([i for i in range(len(indices)) if not torch.any(torch.isnan(tgt_vecs[i]))])
        negs_no_nan_indices = set([i for i in range(len(indices)) if not torch.any(torch.isnan(neg_sense_vecs[i]))])
        no_nan_indices = src_no_nan_indices & tgt_no_nan_indices & negs_no_nan_indices

        # sort
        no_nan_indices = sorted(list(no_nan_indices))

        src_vecs = [src_vecs[i][self.bottom:self.top, :].cpu().numpy() for i in no_nan_indices]
        tgt_vecs = [tgt_vecs[i][self.bottom:self.top, :].cpu().numpy() for i in no_nan_indices]
        neg_sense_vecs = [neg_sense_vecs[i][self.bottom:self.top, :].cpu().numpy() for i in
                          no_nan_indices]

        return src_vecs, tgt_vecs, neg_sense_vecs


def load_corpus(path):
    targets, sents = [], []
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')
            targets.append(array[0])
            sents.append(array[1])

    return targets, sents


def load_corpora():
    # Read paraphrase corpora
    targets, sents = load_corpus(args.data)
    positive_targets, positive_sents = load_corpus(args.data.replace('.and', '.positive_and'))
    neg_sense_targets, neg_sense_sents = load_corpus(args.data.replace('.and', '.negative_sense_and'))

    return targets, sents, positive_targets, positive_sents, neg_sense_targets, neg_sense_sents


def main():
    corpus_name = os.path.basename(args.data)
    out_paths = [args.out + corpus_name + '.' + args.bert,
                 args.out + corpus_name.replace('.and', '.positive_and') + '.' + args.bert,
                 args.out + corpus_name.replace('.and', '.negative_sense_and') + '.' + args.bert]

    targets, sents, positive_targets, positive_sents, neg_sense_targets, neg_sense_sents = load_corpora()
    embedder = Embedder(args.bottom, args.top, targets, sents, positive_targets, positive_sents, neg_sense_targets,
                        neg_sense_sents)

    # Prepare dataloader
    src_vec_buffer, tgt_vec_buffer, neg_vec_buffer = [], [], []
    batch_sampler = BatchSampler(SequentialSampler(range(len(targets))), batch_size=args.batch, drop_last=False)
    for indices in tqdm(batch_sampler):
        src_vecs, tgt_vecs, neg_sense_vecs = embedder.encode_by_bert(indices)
        src_vec_buffer += src_vecs
        tgt_vec_buffer += tgt_vecs
        neg_vec_buffer += neg_sense_vecs

    for path, vecs in zip(out_paths, [src_vec_buffer, tgt_vec_buffer, neg_vec_buffer]):
        np.savez_compressed(path, vecs=np.array(vecs))


if __name__ == '__main__':
    main()
