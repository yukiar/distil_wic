import os, argparse, codecs, glob
import torch
import numpy as np
from pretrained_model import Pretrained_Model_for_Sent
from distiller import Distiller
from sts_unsupervised import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STS17Eval
from util import Cosine
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Evaluate on STS.')
parser.add_argument('--bert', help='pre-trained model name', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
                             'xlm-roberta-base', 'xlm-roberta-large', 'sentence-transformers', 'labse', 'bert-base-multilingual-cased'])
parser.add_argument('--layer', nargs='+', help='layer to use', type=int, default=12)
parser.add_argument('--model_path', help='Directory of checkpoints', type=str)
parser.add_argument('--target', help='Evaluation target', type=str,
                    choices=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17-en-ar', 'STS17-en-de', 'STS17-en-tr',
                             'STS17-es-en', 'STS17-fr-en', 'STS17-it-en', 'STS17-nl-en'],
                    required=True)
parser.add_argument('--out', help='Output directry', type=str, required=True)
parser.add_argument('--batch_size', help='batch size', type=int, default=64)
parser.add_argument('--verbose', help='Output predictions', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    out_dir = args.out
    file_prefix = args.bert
    if args.model_path is not None and 'mim' in args.model_path:
        file_prefix = 'LiuEtAl2020'

    layer_label = '_layer-{0}-{1}_'.format(args.layer[0], args.layer[-1]) if len(
        args.layer) > 1 else 'layer-{0}'.format(args.layer[0])
    out_path = os.path.join(out_dir, args.target, file_prefix + layer_label + 'disentangle_word-sim.txt')

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if args.target.startswith('STS17'):
        sts = STS17Eval('../data/STS/' + args.target)
    else:
        fpath = args.target + '-en-test'
        sts = eval(args.target + 'Eval')('../data/STS/' + fpath)

    # Load pre-trained model
    premodel = Pretrained_Model_for_Sent(args.bert, args.layer, max_length=512)
    upper_layer = int(np.ceil(premodel.layer_num / 2))
    ae_disentangler, mapping = None, None

    sim_measures = ['baseline_MeanCos']

    if args.model_path is not None:
        # AE discentangler
        ae_model_path = glob.glob(os.path.join(args.model_path, '**/', 'ae-*.ckpt'), recursive=True)
        if len(ae_model_path) > 0:
            sim_measures += ['Sense_MeanCos', 'Context_MeanCos']
            ae_model_path = ae_model_path[0]

            ae_disentangler = Distiller.load_from_checkpoint(ae_model_path)
            ae_disentangler.to('cuda')
            ae_disentangler.eval()

    # Evaluation records
    results = {}
    manual_dict, auto_dict = {}, {}
    # init auto_dict
    for sim_measure in sim_measures:
        results[sim_measure] = {}
        auto_dict[sim_measure] = {}

    for dataset in sts.datasets:
        manual_dict[dataset] = []
        for sim_measure in sim_measures:
            auto_dict[sim_measure][dataset] = []

        input1, input2, gs_scores = sts.data[dataset]
        for ii in range(0, len(gs_scores), args.batch_size):
            batch1 = [' '.join(l) for l in input1[ii:ii + args.batch_size]]
            batch2 = [' '.join(l) for l in input2[ii:ii + args.batch_size]]
            batch_gs = gs_scores[ii:ii + args.batch_size]

            s_vecs, s_attn_mask, _ = premodel.encode(batch1)  # [Tensor(len, layer, emb)]
            t_vecs, t_attn_mask, _ = premodel.encode(batch2)  # [Tensor(len, layer, emb)]

            for b in range(len(batch1)):
                manual_dict[dataset].append(batch_gs[b])

                s_vecs_in = s_vecs[b][:, upper_layer:][s_attn_mask[b] == 1] # Skip padding
                t_vecs_in = t_vecs[b][:, upper_layer:][t_attn_mask[b] == 1] # Skip padding
                s_vecs_in_mean = s_vecs_in.mean(axis=1)
                t_vecs_in_mean = t_vecs_in.mean(axis=1)
                mean_cos = Cosine(s_vecs_in_mean, t_vecs_in_mean)
                auto_dict['baseline_MeanCos'][dataset].append(mean_cos)

                # AE
                if ae_disentangler is not None:
                    with torch.no_grad():
                        s_sense, s_context = ae_disentangler(s_vecs_in)
                        t_sense, t_context = ae_disentangler(t_vecs_in)

                    mean_cos = Cosine(s_sense, t_sense)
                    auto_dict['Sense_MeanCos'][dataset].append(mean_cos)

                    mean_cos = Cosine(s_context, t_context)
                    auto_dict['Context_MeanCos'][dataset].append(mean_cos)

        for sim_measure in sim_measures:
            prs = pearsonr(auto_dict[sim_measure][dataset], manual_dict[dataset])[0]
            spr = spearmanr(auto_dict[sim_measure][dataset], manual_dict[dataset])[0]
            results[sim_measure][dataset] = {'pearson': prs, 'spearman': spr, 'nsamples': len(manual_dict[dataset])}
            print('%s %s : pearson = %.4f, spearman = %.4f' % (dataset, sim_measure, prs, spr))
            with open(out_path, 'a') as fw:
                fw.write('{0}\t{1}\t{2}\t{3}\n'.format(dataset, sim_measure, prs, spr))

    for sim_measure in sim_measures:
        weights = [results[sim_measure][dset]['nsamples'] for dset in sts.datasets]
        list_prs = np.array([results[sim_measure][dset]['pearson'] for dset in sts.datasets])
        list_spr = np.array([results[sim_measure][dset]['spearman'] for dset in sts.datasets])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)

        print(
            'ALL (weighted average) %s : Pearson = %.4f, Spearman = %.4f' % (sim_measure, wavg_pearson, wavg_spearman))
        print('ALL (average) %s : Pearson = %.4f, Spearman = %.4f' % (sim_measure, avg_pearson, avg_spearman))

        with open(out_path, 'a') as fw:
            fw.write('ALL (weighted average)\t{0}\t{1}\t{2}\n'.format(sim_measure, wavg_pearson, wavg_spearman))
            fw.write('ALL (average)\t{0}\t{1}\t{2}\n'.format(sim_measure, avg_pearson, avg_spearman))

        if args.verbose:
            # Output Pearson correlations: Cos
            if '_MeanCos' in sim_measure:
                out_path_prediction = os.path.join(out_dir, args.target + '_Cos',
                                                   file_prefix + layer_label + '_' + sim_measure + '_pred.txt')
                if not os.path.exists(os.path.dirname(out_path_prediction)):
                    os.makedirs(os.path.dirname(out_path_prediction))

                with open(out_path_prediction, 'a') as fw:
                    for r in list_prs:
                        fw.write('{0}\n'.format(r))
