import os, argparse, codecs, glob, json, random
import torch
import numpy as np
from tqdm import tqdm
from pretrained_model import Pretrained_Model
from distiller import Distiller
from util import smart_lower, sort_by_idx, compute_accuracy
import warnings
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Evaluate on WiC.')
parser.add_argument('--bert', help='pre-trained model name', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
                             'xlm-roberta-base', 'xlm-roberta-large', 'sentence-transformers', 'labse',
                             'bert-base-multilingual-cased'])
parser.add_argument('--layer', nargs='+', help='layer to use', type=int, default=12)
parser.add_argument('--model_path', help='Directory of checkpoints', type=str)
parser.add_argument('--target', help='Evaluation target', type=str,
                    choices=['SCWS', 'WiC', 'CoSimLex', 'USim', 'MCL-WiC-en-ar', 'MCL-WiC-en-fr', 'MCL-WiC-en-ru',
                             'MCL-WiC-en-zh'],
                    required=True)
parser.add_argument('--out', help='Output directry', type=str, required=True)
parser.add_argument('--verbose', help='Output predictions', action='store_true')
args = parser.parse_args()


class WiC:
    def __init__(self, w1, w2, c1, c2, label):
        self.w1 = smart_lower(w1)
        self.w2 = smart_lower(w2)
        self.c1 = c1
        self.c2 = c2
        self.label = label

    def add_label(self, label):
        self.label = label

    def __str__(self):
        return 'w1:' + self.w1 + ' w2:' + self.w2 + ' c1:' + self.c1 + ' c2:' + self.c2 + ' l:' + str(self.label)


def load_eval_dataset():
    train, dev, test = [], [], []
    if args.target == 'SCWS':
        path = '../data/SCWS/ratings_processed_allcontext.txt'
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                a = [x.strip() for x in line.split('\t')]
                item = WiC(a[1], a[3], a[5], a[6], float(a[7]))
                dev.append(item)
    elif args.target == 'WiC':
        dir = '../data/WiC/'
        for type in ['dev', 'test']:
            with codecs.open(os.path.join(dir, type, type + '.data.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    a = [x.strip() for x in line.split('\t')]
                    indices = [int(idx) for idx in a[2].split('-')]
                    s1 = a[3].strip().split(' ')
                    s2 = a[4].strip().split(' ')
                    item = WiC(s1[indices[0]], s2[indices[1]], a[3], a[4], None)
                    ''' 
                    Special treatment for "'ve" in WiC test set (apostrophe is divided by BERT tokenizer)
                    '''
                    if item.w2 == "\'ve":
                        item.w2 = "ve"

                    eval(type).append(item)
            if type != 'test':
                with codecs.open(os.path.join(dir, type, type + '.gold.txt'), 'r', encoding='utf-8') as f:
                    idx = 0
                    for line in f:
                        eval(type)[idx].add_label(line.strip())
                        idx += 1
    elif args.target == 'CoSimLex':
        path = '../data/CoSimLex/cosimlex_en_processed_allcontext.txt'  # Preprocessed cosimlex
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                a = [x.strip() for x in line.split('\t')]
                # context 1
                item = WiC(a[11], a[12], a[2], a[3], float(a[6]))
                dev.append(item)
                # context 2
                item = WiC(a[13], a[14], a[4], a[5], float(a[7]))
                dev.append(item)
    elif args.target == 'USim':
        path = '../data/USim/usim_preprocessed.txt'
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                a = [x.strip() for x in line.split('\t')]
                item = WiC(a[0], a[1], a[2], a[3], float(a[4]))
                dev.append(item)
    elif args.target.startswith('MCL-WiC'):
        # Use corresponding development datasets in languages other than English
        tlang = args.target[-5:].split('-')[1]
        lang_pair = tlang + '-' + tlang
        path = '../data/MCL-WiC/MCL-WiC-' + lang_pair + '/dev/' + 'dev.' + lang_pair + '.processed.tsv'
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                a = [x.strip() for x in line.split('\t')]
                item = WiC(a[0], a[1], a[2], a[3], a[4])
                dev.append(item)

        lang_pair = args.target[-5:]
        path = '../data/MCL-WiC/' + args.target + '/test/' + 'test.' + lang_pair + '.processed.tsv'
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                a = [x.strip() for x in line.split('\t')]
                item = WiC(a[0], a[1], a[2], a[3], a[4])
                test.append(item)

    return train, dev, test



if __name__ == '__main__':
    out_dir = args.out
    file_prefix = args.bert
    layer_label = '_layer-{0}-{1}_'.format(args.layer[0], args.layer[-1]) if len(
        args.layer) > 1 else 'layer-{0}'.format(args.layer[0])
    out_path = os.path.join(out_dir, args.target, file_prefix + layer_label + 'disentangle_word-sim.txt')

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    train, dev, test = load_eval_dataset()

    # Load pre-trained model
    premodel = Pretrained_Model(args.bert, args.layer, max_length=512)
    upper_layer = int(np.ceil(premodel.layer_num / 2))
    vae_disentangler, pre_vae_disentangler, ae_disentangler = None, None, None

    sim_measures = ['baseline_context-cossim']

    if args.model_path is not None:
        # AE discentangler
        ae_model_path = glob.glob(os.path.join(args.model_path, '**/', 'ae-*.ckpt'), recursive=True)
        if len(ae_model_path) > 0:
            sim_measures += ['AECosS', 'AECosC']
            ae_model_path = ae_model_path[0]

            ae_disentangler = Distiller.load_from_checkpoint(ae_model_path)
            ae_disentangler.to('cuda')
            ae_disentangler.eval()

    # WiC best threshold
    wic_best_acccuracy, wic_best_thresh = {}, {}
    for sim_measure in sim_measures:
        wic_best_acccuracy[sim_measure] = 0
        wic_best_thresh[sim_measure] = 0

    for eval_type in ['dev', 'test']:
        dataset = eval(eval_type)
        if len(dataset) == 0:
            continue

        # Evaluation records
        manual_dict, auto_dict = {}, {}
        not_found, total_size = (0, 0)
        # init auto_dict
        for sim_measure in sim_measures:
            auto_dict[sim_measure] = {}

        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            # Encode word in contexts
            item = dataset[i]
            w1_in_context = torch.stack(premodel.encode([item.w1], [[item.c1]], mean=False)).detach().to('cuda')
            w2_in_context = torch.stack(premodel.encode([item.w2], [[item.c2]], mean=False)).detach().to('cuda')

            if torch.any(torch.isnan(w1_in_context)) or torch.any(torch.isnan(w2_in_context)):
                not_found += 1
            else:
                w1_in_context = w1_in_context[:, upper_layer:]
                w2_in_context = w2_in_context[:, upper_layer:]
                w1_in_context_mean = w1_in_context.mean(axis=1).to('cpu').numpy()
                w2_in_context_mean = w2_in_context.mean(axis=1).to('cpu').numpy()

                # ground-truth
                manual_dict[i] = item.label

                # baseline: NO clustering, mean of all pairs of cosine distances
                auto_dict['baseline_context-cossim'][i] = cosine_similarity(w1_in_context_mean, w2_in_context_mean)[
                    0, 0]

                # AE
                if ae_disentangler is not None:
                    with torch.no_grad():
                        w1_sense, w1_context = ae_disentangler(w1_in_context)
                        w2_sense, w2_context = ae_disentangler(w2_in_context)

                    w1_sense = w1_sense.cpu().numpy()
                    w1_context = w1_context.cpu().numpy()
                    w2_sense = w2_sense.cpu().numpy()
                    w2_context = w2_context.cpu().numpy()
                    ae_sense_cos = cosine_similarity(w1_sense, w2_sense)[0, 0]
                    auto_dict['AECosS'][i] = ae_sense_cos
                    ae_context_cos = cosine_similarity(w1_context, w2_context)[0, 0]
                    auto_dict['AECosC'][i] = ae_context_cos

            total_size += 1

        if args.target.startswith('MCL-WiC'):
            # Set threshold on en-en dev set
            if eval_type == 'dev':
                indices = list(manual_dict.keys())
                for th in np.arange(1.0, -1.01, -0.001):
                    for sim_measure, sim_dict in auto_dict.items():
                        binarized_autodict = {}
                        for idx in indices:
                            binarized_autodict[idx] = 'T' if sim_dict[idx] >= th else 'F'

                        accuracy = compute_accuracy(manual_dict, binarized_autodict)
                        if wic_best_acccuracy[sim_measure] < accuracy:
                            wic_best_thresh[sim_measure] = th
                            wic_best_acccuracy[sim_measure] = accuracy
            else:
                indices = list(manual_dict.keys())
                test_accuracies = {}
                for sim_measure, sim_dict in auto_dict.items():
                    binarized_predictions = {}
                    # Convert predictions to 'T' or 'F'
                    for idx in indices:
                        binarized_predictions[idx] = 'T' if sim_dict[idx] >= wic_best_thresh[sim_measure] else 'F'

                    accuracy = compute_accuracy(manual_dict, binarized_predictions)
                    test_accuracies[sim_measure] = accuracy

                with open(out_path, 'a') as fw:
                    print(
                        '============================================= RESULTS ============================================================')
                    print("%20s" % "Sim Measure", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Threshold",
                          "%15s" % "Dev Accuracy", "%15s" % "Test Accuracy")
                    print(
                        '==================================================================================================================')
                    fw.write(
                        '============================================= RESULTS ============================================================\n')
                    fw.write('Sim Measure\tNum Pairs\tNot found\tThreshold\tDev Accuracy\tTest Accuracy\n')
                    fw.write(
                        '===================================================================================================================\n')

                    for sim_measure, accuracy in test_accuracies.items():
                        print("%20s" % sim_measure,
                              "%15s" % str(total_size),
                              "%15s" % str(not_found),
                              "%15.3f" % wic_best_thresh[sim_measure],
                              "%15.3f" % wic_best_acccuracy[sim_measure],
                              "%15.3f" % accuracy)
                        fw.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(sim_measure, total_size, not_found,
                                                                         wic_best_thresh[sim_measure],
                                                                         wic_best_acccuracy[sim_measure], accuracy))

        elif args.target == 'WiC':
            if eval_type == 'dev':
                print(
                    '=======================================================================================================================')
                print("%20s" % "Eval Type", "%15s" % "Threshold", "%20s" % "Sim Measure", "%15s" % "Num Pairs",
                      "%15s" % "Not found", "%15s" % "Accuracy")
                print(
                    '=======================================================================================================================')
                with open(out_path, 'a') as fw:
                    fw.write(
                        'Eval Type\tThreshold\tSim Measure\tNum Pairs\tNot found\tAccuracy\n')

                # Binalize continuous similarity
                for th in np.arange(1.0, -1.01, -0.001):
                    binarized_autodict = {}
                    for sim_measure, sim_dict in auto_dict.items():
                        binarized_autodict[sim_measure] = {}
                        for k, v in auto_dict[sim_measure].items():
                            binarized_autodict[sim_measure][k] = 'T' if v >= th else 'F'
                    for sim_measure, sim_dict in binarized_autodict.items():
                        accuracy = compute_accuracy(manual_dict, sim_dict)
                        if wic_best_acccuracy[sim_measure] < accuracy:
                            wic_best_thresh[sim_measure] = th
                            wic_best_acccuracy[sim_measure] = accuracy

                with open(out_path, 'a') as fw:
                    for sim_measure in sim_measures:
                        print("%20s" % args.target + '-' + eval_type, "%15.3f" % wic_best_thresh[sim_measure],
                              "%20s" % sim_measure,
                              "%15s" % str(total_size),
                              "%15s" % str(not_found),
                              "%15.3f" % wic_best_acccuracy[sim_measure])
                        fw.write(
                            '{0}\t{1:.3f}\t{2}\t{3}\t{4}\t{5:.3f}\n'.format(
                                args.target + '-' + eval_type, wic_best_thresh[sim_measure],
                                sim_measure,
                                total_size,
                                not_found,
                                wic_best_acccuracy[sim_measure]))

            else:  # Create submission file for CodaLab
                print(
                    '=======================================================================================================================')
                print("%20s" % "Eval Type", "%15s" % "Threshold", "%20s" % "Sim Measure", "%15s" % "Num Pairs",
                      "%15s" % "Not found")
                print(
                    '=======================================================================================================================')
                with open(out_path, 'a') as fw:
                    fw.write(
                        'Eval Type\tThreshold\tSim Measure\tNum Pairs\tNot found\n')

                binarized_autodict = {}
                best_measure = max(wic_best_acccuracy, key=wic_best_acccuracy.get)
                for sim_measure in sim_measures:
                    binarized_autodict[sim_measure] = {}
                    # Convert predictions to 'T' or 'F'
                    for k, v in auto_dict[sim_measure].items():
                        binarized_autodict[sim_measure][k] = 'T' if v >= wic_best_thresh[sim_measure] else 'F'

                    # Output
                    if sim_measure == best_measure:
                        out_path_prediction = out_path.replace('word-sim.txt',
                                                               '_') + 'BEST_' + sim_measure + '_{0:.3f}'.format(
                            wic_best_thresh[sim_measure]) + '_output.txt'
                    else:
                        out_path_prediction = out_path.replace('word-sim.txt', '_') + sim_measure + '_{0:.3f}'.format(
                            wic_best_thresh[sim_measure]) + '_output.txt'
                    with open(out_path_prediction, 'w') as fw:
                        for i in range(len(test)):
                            if i in binarized_autodict[sim_measure]:
                                fw.write(binarized_autodict[sim_measure][i] + '\n')
                            else:  # OOV
                                fw.write('F\n')

                # Statistics
                with open(out_path, 'a') as fw:
                    for sim_measure, sim_dict in binarized_autodict.items():
                        print("%20s" % args.target + '-' + eval_type, "%15.3f" % wic_best_thresh[sim_measure],
                              "%20s" % sim_measure,
                              "%15s" % str(total_size),
                              "%15s" % str(not_found))
                        fw.write(
                            '{0}\t{1:.3f}\t{2}\t{3}\t{4}\n'.format(args.target + '-' + eval_type,
                                                                   wic_best_thresh[sim_measure],
                                                                   sim_measure,
                                                                   total_size,
                                                                   not_found))

        elif args.target == 'CoSimLex':
            ### CoSimLex Task 1
            manual_dict_I, auto_dict_I = {}, {}
            for sim_measure in sim_measures:
                auto_dict_I[sim_measure] = {}
            for k in manual_dict.keys():
                if k % 2 == 0 and (k + 1) in manual_dict:
                    manual_dict_I[k] = manual_dict[k + 1] - manual_dict[k]
                    for sim_measure in sim_measures:
                        auto_dict_I[sim_measure][k] = auto_dict[sim_measure][k + 1] - auto_dict[sim_measure][k]
            print(
                '=======================================================================================================================')
            print("%20s" % "Eval Type", "%20s" % "Sim Measure", "%15s" % "Num Pairs", "%15s" % "Not found",
                  "%15s" % "r")
            print(
                '=======================================================================================================================')
            with open(out_path, 'a') as fw:
                fw.write('Eval Type\tSim Measure\tNum Pairs\tNot found\tr\n')
            with open(out_path, 'a') as fw:
                for sim_measure, sim_dict in auto_dict_I.items():
                    scipy_r = pearsonr(sort_by_idx(manual_dict_I), sort_by_idx(sim_dict))
                    print("%20s" % args.target + ' Task-I', "%20s" % sim_measure, "%15s" % str(total_size),
                          "%15s" % str(not_found), "%15.4f" % scipy_r[0])
                    fw.write(
                        '{0}\t{1}\t{2}\t{3}\t{4:.4f}\n'.format(args.target + ' Task-I', sim_measure,
                                                               total_size,
                                                               not_found, scipy_r[0]))

            ### CoSimLex Task 2
            print(
                '=======================================================================================================================')
            print("%20s" % "Eval Type", "%20s" % "Sim Measure", "%15s" % "Num Pairs", "%15s" % "Not found",
                  "%15s" % "Rho")
            print(
                '=======================================================================================================================')
            with open(out_path, 'a') as fw:
                fw.write('Eval Type\tSim Measure\tNum Pairs\tNot found\tRho\n')
            with open(out_path, 'a') as fw:
                for sim_measure, sim_dict in auto_dict.items():
                    scipy_rho = spearmanr(sort_by_idx(manual_dict), sort_by_idx(sim_dict))
                    print("%20s" % args.target + ' Task-II', "%20s" % sim_measure, "%15s" % str(total_size),
                          "%15s" % str(not_found),
                          "%15.4f" % scipy_rho[0])
                    fw.write(
                        '{0}\t{1}\t{2}\t{3}\t{4:.4f}\n'.format(args.target + ' Task-II', sim_measure,
                                                               total_size,
                                                               not_found, scipy_rho[0]))
        else:
            print(
                '=======================================================================================================================')
            print("%20s" % "Eval Type", "%20s" % "Sim Measure", "%15s" % "Num Pairs", "%15s" % "Not found",
                  "%15s" % "Rho")
            print(
                '=======================================================================================================================')
            with open(out_path, 'a') as fw:
                fw.write('Eval Type\tSim Measure\tNum Pairs\tNot found\tRho\n')
            with open(out_path, 'a') as fw:
                for sim_measure, sim_dict in auto_dict.items():
                    scipy_rho = spearmanr(sort_by_idx(manual_dict), sort_by_idx(sim_dict))
                    print("%20s" % args.target, "%20s" % sim_measure, "%15s" % str(total_size), "%15s" % str(not_found),
                          "%15.4f" % scipy_rho[0])
                    fw.write(
                        '{0}\t{1}\t{2}\t{3}\t{4:.4f}\n'.format(args.target, sim_measure, total_size,
                                                               not_found, scipy_rho[0]))

        if args.verbose and eval_type == 'dev':
            # Output predictions
            if args.target == 'CoSimLex':
                # Output predictions for CoSimLexI
                for sim_measure, sim_dict in auto_dict.items():
                    out_path_prediction = os.path.join(out_dir, 'CoSimLexI',
                                                       file_prefix + layer_label + 'disentangle_' + sim_measure + '_' + eval_type + '_pred.txt')
                    if not os.path.exists(os.path.dirname(out_path_prediction)):
                        os.makedirs(os.path.dirname(out_path_prediction))
                    auto_dict_I = {}
                    manual_dict_I = {}
                    for k in sim_dict.keys():
                        if k % 2 == 0 and (k + 1) in sim_dict:
                            manual_dict_I[k] = manual_dict[k + 1] - manual_dict[k]
                            auto_dict_I[k] = sim_dict[k + 1] - sim_dict[k]
                    predictions = [i for i in sort_by_idx(auto_dict_I)]
                    with open(out_path_prediction, 'w') as fw:
                        for pred in predictions:
                            fw.write('{0}\n'.format(pred))
                # Output gold for CoSimLexI
                out_path_gold = os.path.join(out_dir, 'CoSimLexI', 'gold.txt')
                with open(out_path_gold, 'w') as fw:
                    for gold in sort_by_idx(manual_dict_I):
                        fw.write('{0}\n'.format(gold))
            elif args.target == 'WiC' or args.target.startswith('MCL-WiC'):
                binarized_autodict = {}
                for sim_measure in sim_measures:
                    binarized_autodict[sim_measure] = {}
                    # Convert predictions to 'T' or 'F'
                    for k, v in auto_dict[sim_measure].items():
                        binarized_autodict[sim_measure][k] = 'T' if v >= wic_best_thresh[sim_measure] else 'F'
                auto_dict = binarized_autodict

            for sim_measure, sim_dict in auto_dict.items():
                out_path_prediction = out_path.replace('word-sim.txt',
                                                       '_') + sim_measure + '_' + eval_type + '_pred.txt'
                predictions = [i for i in sort_by_idx(sim_dict)]

                with open(out_path_prediction, 'w') as fw:
                    for pred in predictions:
                        fw.write('{0}\n'.format(pred))

            # Output gold
            out_path_gold = os.path.join(out_dir, args.target, 'gold.txt')
            with open(out_path_gold, 'w') as fw:
                for gold in sort_by_idx(manual_dict):
                    fw.write('{0}\n'.format(gold))
