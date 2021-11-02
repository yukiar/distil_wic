from transformers import AutoTokenizer, AutoModelForMaskedLM
from util import tokenize
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

class BERTMaskPrediction:
    padding = 'do_not_pad'

    def __init__(self, multilingual=False, max_length=512):
        self.max_length = max_length
        self.space_symbol, self.subword_symbol = None, None
        if multilingual:
            model_type = 'xlm-roberta-large'
            self.subword_symbol = self.space_symbol = '▁'
        else:
            model_type = 'bert-large-cased-whole-word-masking'
            self.subword_symbol = self.subword_symbol = '##'

        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = AutoModelForMaskedLM.from_pretrained(model_type).cuda()

    def softmax(self, x):
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

    def guess_masked_token(self, vocab, sent, target, topK, probT):
        final_pred_tokens, final_pred_probs = [], []
        input_ids, _, term_idxes = tokenize(self.tokenizer, self.space_symbol, self.subword_symbol,
                                            self.padding, self.max_length,
                                            True, False, False, target,
                                            [sent], None)
        input_ids = input_ids[0]
        term_idxes = term_idxes[0]
        to_mask_indices = [i for i, x in enumerate(term_idxes) if x == 1]

        if len(to_mask_indices) == 0:  # matching failed
            return final_pred_tokens, final_pred_probs

        for i in to_mask_indices:
            input_ids[i] = self.tokenizer.mask_token_id

        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.cuda()

        all_pred_subwords, all_sw_pred_probs = [], []
        with torch.no_grad():
            preds = self.model(tens)[0]
            probs = self.softmax(preds)

            for target_idx in to_mask_indices:
                pred_top = torch.topk(probs[0, target_idx], topK)
                pred_prob = pred_top[0].tolist()
                pred_idx = pred_top[1].tolist()

                pred_tok = self.tokenizer.convert_ids_to_tokens(pred_idx)
                all_pred_subwords.append(pred_tok)
                all_sw_pred_probs.append(pred_prob)

        # Concat subwords
        for k in range(topK):
            token = [all_pred_subwords[i][k] for i in range(len(to_mask_indices))]
            prob_all = [all_sw_pred_probs[i][k] for i in range(len(to_mask_indices))]
            # remove subword symbols
            token = self.tokenizer.convert_tokens_to_string(token)
            prob = sum(prob_all) / len(prob_all)

            if prob > probT:
                if target.lower() != token.lower():
                    if len(token) > 0 and len(token.split(' ')) == 1:
                        if token.lower() in vocab:  # Exclude FastText OOV
                            final_pred_tokens.append(token)
                            final_pred_probs.append(prob)
            else:
                break
        return final_pred_tokens, final_pred_probs

