import torch, copy
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, AutoModel
import numpy as np
from util import tokenize

np.random.seed(42)
torch.manual_seed(42)


class Pretrained_Model:
    ######## Setting ##########
    padding = 'max_length'
    truncation = True
    ambiguous_match = False

    #################
    def __init__(self, model_ver, layer, device='cuda', max_length=512):  # input sentences are 5-30 words
        super().__init__()
        if '-' in layer:
            min_max_layer = [int(l) for l in layer.split('-')]
            self.layer = [i for i in range(min_max_layer[0], min_max_layer[1] + 1)]
        elif type(layer) is list:
            self.layer = layer
        else:
            self.layer = [int(layer)]

        self.device = device
        self.layer_num = len(layer)
        self.max_length = max_length
        self.tokenizer, self.model = None, None
        self.output_hidden_index = 1 if 'distilbert' in model_ver else 2
        self.space_symbol, self.subword_symbol = None, None

        if 'roberta' in model_ver:
            self.space_symbol = '▁'
            self.tokenizer = AutoTokenizer.from_pretrained(model_ver)
        elif 'bert' in model_ver:
            self.subword_symbol = '##'
            self.tokenizer = AutoTokenizer.from_pretrained(model_ver, do_basic_tokenize=False)
        elif 'sentence-transformers' in model_ver:
            self.space_symbol = '▁'
            model_ver = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
            self.tokenizer = AutoTokenizer.from_pretrained(model_ver)
        else:
            raise TypeError('{0} is unavailable.'.format(model_ver))

        self.model = AutoModel.from_pretrained(model_ver)
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.model.eval()
        self.model.to(self.device)

    def encode_with_mask(self, words, sample_sentence_list, match_all=True, mean=True, logger=None):
        input_ids_buffer, atten_mask_buffer, term_idxes_buffer = [], [], []
        for word, sample_sentences in zip(words, sample_sentence_list):
            input_ids, attention_mask, term_idxes = tokenize(self.tokenizer, self.space_symbol, self.subword_symbol,
                                                             self.padding, self.max_length,
                                                             self.truncation, match_all, self.ambiguous_match, word,
                                                             sample_sentences, logger)
            for i_id, a_mask, term_mask in zip(input_ids, attention_mask, term_idxes):
                t_indices = [i for i, x in enumerate(term_mask) if x == 1]
                i_masked_idx = copy.deepcopy(i_id)
                term_except_mask = copy.deepcopy(a_mask)
                for i in t_indices:
                    i_masked_idx[i] = self.tokenizer.mask_token_id
                    term_except_mask[i] = 0

                # for normal embeddings
                input_ids_buffer.append(i_id)
                atten_mask_buffer.append(a_mask)
                term_idxes_buffer.append(term_mask)

                # for masked embeddings
                input_ids_buffer.append(i_masked_idx)
                # input_ids_buffer.append(i_id)
                atten_mask_buffer.append(a_mask)
                # # Use [MASK] embedding
                # term_idxes_buffer.append(term_mask)
                # # Use [CLS] embedding
                # term_mask_cls = [0] * len(term_mask)
                # term_mask_cls[0] = 1
                # term_idxes_buffer.append(term_mask_cls)
                # Use Average of all takens except [Mask]
                term_idxes_buffer.append(term_except_mask)

        # Convert to Tensors and build dataset
        input_ids = torch.tensor(input_ids_buffer, dtype=torch.long)
        input_mask = torch.tensor(atten_mask_buffer, dtype=torch.long)
        term_idxes = torch.tensor(term_idxes_buffer, dtype=torch.long)
        all_example_index = torch.arange(input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            input_ids, input_mask, term_idxes, all_example_index
        )

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=input_ids.shape[0])

        embeddings = []
        batch = next(iter(dataloader))
        batch = tuple(t.to(self.model.device) for t in batch)
        # Predict hidden states features for each layer
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            # See the models docstrings for the detail of the inputs
            outputs = self.model(**inputs, output_hidden_states=True)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            # encoded_layers = outputs[0]
            for b in range(outputs[0].size()[0]):  # Compute mean of term embeddings for subword segmentation
                token_embs = self.get_token_embeddings(outputs, b, batch[2][b], mean)
                embeddings.append(token_embs.squeeze())

                # embeddings.append(
                #     torch.mean(outputs[self.output_hidden_index][self.layer][b, torch.nonzero(batch[2][b])],
                #                dim=0).squeeze())

        return embeddings

    def encode(self, words, sample_sentence_list, match_all=True, mean=True, logger=None):
        input_ids_buffer, atten_mask_buffer, term_idxes_buffer = [], [], []
        for word, sample_sentences in zip(words, sample_sentence_list):
            input_ids, attention_mask, term_idxes = tokenize(self.tokenizer, self.space_symbol, self.subword_symbol,
                                                             self.padding, self.max_length,
                                                             self.truncation, match_all, self.ambiguous_match, word,
                                                             sample_sentences, logger)
            input_ids_buffer += input_ids
            atten_mask_buffer += attention_mask
            term_idxes_buffer += term_idxes

        # Convert to Tensors and build dataset
        input_ids = torch.tensor(input_ids_buffer, dtype=torch.long)
        input_mask = torch.tensor(atten_mask_buffer, dtype=torch.long)
        term_idxes = torch.tensor(term_idxes_buffer, dtype=torch.long)
        all_example_index = torch.arange(input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            input_ids, input_mask, term_idxes, all_example_index
        )

        dataloader = DataLoader(dataset, batch_size=input_ids.shape[0])

        embeddings = []
        batch = next(iter(dataloader))
        batch = tuple(t.to(self.model.device) for t in batch)
        # Predict hidden states features for each layer
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            # See the models docstrings for the detail of the inputs
            outputs = self.model(**inputs, output_hidden_states=True)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            # encoded_layers = outputs[0]
            for b in range(outputs[0].size()[0]):  # Compute mean of term embeddings for subword segmentation
                token_embs = self.get_token_embeddings(outputs, b, batch[2][b], mean)
                embeddings.append(token_embs.squeeze())

                # embeddings.append(
                #     torch.mean(outputs[self.output_hidden_index][self.layer][b, torch.nonzero(batch[2][b])],
                #                dim=0).squeeze())

        return embeddings

    def encode_for_train(self, words, sentences, random_ids, random_sentences, random_words, match_all=True, mean=True):
        input_ids_buffer, atten_mask_buffer, term_idxes_buffer = [], [], []
        for word, sent_list in zip(words, sentences):
            input_ids, attention_mask, term_idxes = tokenize(self.tokenizer, self.space_symbol, self.subword_symbol,
                                                             self.padding, self.max_length,
                                                             self.truncation, match_all, self.ambiguous_match, word,
                                                             sent_list)
            input_ids_buffer += input_ids
            atten_mask_buffer += attention_mask
            term_idxes_buffer += term_idxes

        random_term_idx_buffer = []
        for word, sent_list in zip(random_words, random_sentences):
            _, _, term_idxes = tokenize(self.tokenizer, self.space_symbol, self.subword_symbol,
                                        self.padding, self.max_length,
                                        self.truncation, match_all, self.ambiguous_match, word,
                                        sent_list)
            random_term_idx_buffer += term_idxes
        random_term_idx_buffer = torch.tensor(random_term_idx_buffer, dtype=torch.long)

        # Convert to Tensors and build dataset
        input_ids = torch.tensor(input_ids_buffer, dtype=torch.long)
        input_mask = torch.tensor(atten_mask_buffer, dtype=torch.long)
        term_idxes = torch.tensor(term_idxes_buffer, dtype=torch.long)
        all_example_index = torch.arange(input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            input_ids, input_mask, term_idxes, all_example_index
        )

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=input_ids.shape[0])

        src_vecs, tgt_vecs, rand_vecs = [], [], []
        batch = next(iter(dataloader))
        batch = tuple(t.to(self.model.device) for t in batch)
        # Predict hidden states features for each layer
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            # See the models docstrings for the detail of the inputs
            outputs = self.model(**inputs, output_hidden_states=True)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            # encoded_layers = outputs[0]
            for b in range(int(outputs[0].size()[0] / 2)):  # Compute mean of term embeddings for subword segmentation
                token_embs = self.get_token_embeddings(outputs, 2 * b, batch[2][2 * b], mean)
                src_vecs.append(token_embs.squeeze())

                token_embs = self.get_token_embeddings(outputs, 2 * b + 1, batch[2][2 * b + 1], mean)
                tgt_vecs.append(token_embs.squeeze())

                rand_token_embs = self.get_token_embeddings(outputs, random_ids[b], random_term_idx_buffer[b], mean)
                rand_vecs.append(rand_token_embs.squeeze())

        return src_vecs, tgt_vecs, rand_vecs

    def get_token_embeddings(self, outputs, b, term_idxes, mean):
        token_embs = outputs[self.output_hidden_index][self.layer[0]][b, torch.nonzero(term_idxes)]
        if len(self.layer) > 1:
            for l in self.layer[1:]:
                token_embs = torch.cat(
                    (token_embs, outputs[self.output_hidden_index][l][b, torch.nonzero(term_idxes)]), dim=1)

        if mean:
            # Mean over layers
            token_embs = torch.mean(token_embs, dim=1, keepdim=True)
        # Mean over sub-words
        token_embs = torch.mean(token_embs, dim=0, keepdim=True)

        return token_embs


class Pretrained_Model_for_Sent:
    def __init__(self, model_ver, layer, max_length=256):  # input sentences are 5-30 words
        if '-' in layer:
            min_max_layer = [int(l) for l in layer.split('-')]
            self.layer = [i for i in range(min_max_layer[0], min_max_layer[1] + 1)]
        elif type(layer) is list:
            self.layer = layer
        else:
            self.layer = [int(layer)]

        self.layer_num = len(layer)
        self.max_length = max_length
        self.output_hidden_index = 1 if 'distilbert' in model_ver else 2

        if 'sentence-transformers' in model_ver:
            model_ver = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'

        self.tokenizer = AutoTokenizer.from_pretrained(model_ver)
        self.model = AutoModel.from_pretrained(model_ver)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.model.eval()
        self.model.to('cuda')

    def encode(self, bsent):
        with torch.no_grad():
            inputs = self.tokenizer(bsent, padding=True, max_length=self.max_length, truncation=True,
                                    return_tensors="pt").to('cuda')
            outputs = self.model(**inputs, output_hidden_states=True)

        # Convert {layer: (batch, len, emb)} to {Batch: (len, layer, emb)}
        batch_embs = []
        for b in range(int(outputs[0].size()[0])):
            embs = outputs[self.output_hidden_index][self.layer[0]][b].unsqueeze(1)
            if len(self.layer) > 1:
                for l in self.layer[1:]:
                    embs = torch.cat((embs, outputs[self.output_hidden_index][l][b].unsqueeze(1)), dim=1)
            batch_embs.append(embs)

        return batch_embs, inputs['attention_mask'], inputs['input_ids']

    def encode_last_hidden_cls(self, bsent):
        with torch.no_grad():
            inputs = self.tokenizer(bsent, padding=True, max_length=self.max_length, truncation=True,
                                    return_tensors="pt").to('cuda')
            outputs = self.model(**inputs, output_hidden_states=True)

        # Convert {layer: (batch, len, emb)} to {Batch: (len, layer, emb)}
        batch_embs = outputs['last_hidden_state'][:, 0].unsqueeze(1)

        return batch_embs, inputs['attention_mask'], inputs['input_ids']
