import argparse, os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from distiller import Distiller
import numpy as np


parser = argparse.ArgumentParser(description='Distill WiC representation.')
parser.add_argument('--out', help='dataset', type=str, default='../out/')
parser.add_argument('--model', help='BERT model name', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
                             'xlm-roberta-base', 'xlm-roberta-large', 'sentence-transformers',
                             'bert-base-multilingual-cased'])
parser.add_argument('--layer', nargs='+', help='layer to use', type=int, default=12)
parser.add_argument('--nhead', help='number of attention heads', type=int, default=8)
parser.add_argument('--ff_k', help='ff dimention in transformer', type=int, default=4)
parser.add_argument('--neg_sense', help='flag to use negative-sense samples', action='store_true')
parser.add_argument('--neg_context', help='flag to use negative-context samples', action='store_true')
parser.add_argument('--cos_loss', help='flag to use cosine loss', action='store_true')
parser.add_argument('--batch', help='Batch size', type=int, default=8)
parser.add_argument('--val_check_interval', help='Number of steps per validation', type=float, default=1.0)
parser.add_argument('--data', help='dataset', type=str, required=True)
parser.add_argument('--dev_size', help='Size of dev set', type=int, default=10000)
parser.add_argument('--init_lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--find_lr', help='automatic learning rate finder', action='store_true')
parser.add_argument('--lr_sample', help='learning rate', type=int, default=100)
parser.add_argument('--warmup', help='decay factor', type=int, default=0)
parser.add_argument('--gpus', nargs='+', help='gpu to use', type=int, default=0)
parser.add_argument('--resume_from', help='resume training', type=str, default=None)
args = parser.parse_args()


def load_corpus(path):
    targets, sents = [], []
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')
            targets.append(array[0])
            sents.append(array[1])

    return targets, sents


def split_corpus(corpus_size):
    # train/val split
    if int(corpus_size * 0.2) > args.dev_size:
        split = corpus_size - args.dev_size
    else:
        split = int(corpus_size * 0.8)
    return split


def load_corpora():
    if args.data.endswith('.npz'):
        print('Loading src npz...')
        src_vecs = np.load(args.data)['vecs']
        print('Loading tgt npz...')
        positive_vecs = np.load(args.data.replace('.and', '.positive_and'))['vecs']
        negative_vecs = None
        if args.neg_sense:
            print('loading negative npz...')
            negative_vecs = np.load(args.data.replace('.and', '.negative_sense_and'))['vecs']
        print('loading done!')
        split = split_corpus(src_vecs.shape[0])
        return (src_vecs, positive_vecs, negative_vecs), src_vecs.shape[0], split
    else:
        # Read paraphrase corpora
        targets, sents = load_corpus(args.data)
        positive_targets, positive_sents = load_corpus(args.data.replace('.and', '.positive_and'))
        neg_sense_targets, neg_sense_sents = None, None
        neg_context_targets, neg_context_sents = None, None
        if args.neg_sense:
            neg_sense_targets, neg_sense_sents = load_corpus(args.data.replace('.and', '.negative_sense_and'))
        # if args.neg_context:
        #     neg_context_targets, neg_context_sents = load_corpus(args.data.replace('.and', '.negative_context_and'))
        split = split_corpus(len(targets))

        return (targets, sents, positive_targets, positive_sents, neg_sense_targets, neg_sense_sents,
                neg_context_targets, neg_context_sents), len(targets), split


def find_learning_rate(mylogger, model):
    trainer = pl.Trainer(logger=mylogger, default_root_dir=mylogger.log_dir, val_check_interval=args.val_check_interval,
                         log_every_n_steps=10, gpus=args.gpus, accelerator='dp')

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, min_lr=1e-7, max_lr=0.1, early_stop_threshold=None,
                                      num_training=args.lr_sample)

    suggested_lr = lr_finder.suggestion()
    #
    print('suggested lr:' + str(suggested_lr))
    return suggested_lr


def construct_model(dataset, corpus_size, split, lr):
    # max_seq_len = max([len(x.split()) for x in sents + positive_sents]) * 3
    max_seq_len = 200
    model = Distiller(args.model, args.layer, args.nhead, args.neg_sense, args.neg_context, args.cos_loss, lr,
                      args.warmup, args.batch, ff_dim_k=args.ff_k, max_seq_len=max_seq_len)

    model.set_train_corpus(dataset, corpus_size, split)

    return model


def main():
    name = args.model + '_' + os.path.basename(args.data)
    dataset, corpus_size, train_size = load_corpora()
    model = construct_model(dataset, corpus_size, train_size, args.init_lr)
    ##########################################
    # AE Training
    ##########################################
    mylogger = TensorBoardLogger(save_dir=args.out, name=name)
    if args.find_lr:
        suggested_lr = find_learning_rate(mylogger, model)
        model.lr = suggested_lr

    # Set up trainer for AE training
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5,
        patience=15,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='ae-allnli-{epoch:02d}-{val_loss:.8f}',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.resume_from is not None and 'val_loss' in args.resume_from:
        trainer = pl.Trainer(logger=mylogger, val_check_interval=args.val_check_interval, log_every_n_steps=10,
                             callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                             resume_from_checkpoint=args.resume_from, gpus=args.gpus,
                             accelerator='dp')  # if necessary, add 'gradient_clip_val=args.clip'
    else:
        trainer = pl.Trainer(logger=mylogger, val_check_interval=args.val_check_interval, log_every_n_steps=10,
                             callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], gpus=args.gpus,
                             accelerator='dp')  # if necessary, add 'gradient_clip_val=args.clip'

    # Training
    trainer.fit(model)


if __name__ == '__main__':
    main()
