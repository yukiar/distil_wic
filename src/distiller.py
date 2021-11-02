import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
import transformers
from pretrained_model import Pretrained_Model
import numpy as np

np.random.seed(42)
torch.manual_seed(42)


class Encoder(nn.Module):
    def __init__(self, embedding_dim, nhead, dropout, k=4):
        super(Encoder, self).__init__()
        # Same setting with BERT
        self.transformer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward=k * embedding_dim,
                                                      dropout=dropout, activation='gelu')

    def forward(self, x):
        # x has a shape (Batch, Layers, Embedding)
        x = x.transpose(0, 1)  # Transform to (Layers, Batch, Embedding)
        h = self.transformer(x)
        out = h.mean(dim=0)
        return out

class Distiller(pl.LightningModule):
    def __init__(self, model_ver, layer, nhead, neg_sense, neg_context, cos_loss, lr, warmup, batch_size,
                 ff_dim_k=4, max_seq_len=512):
        super().__init__()
        self.save_hyperparameters()

        # Settings
        self.lr = lr
        self.warmup = warmup
        # self.latent_dim = latent_dim
        self.with_neg_sense = neg_sense
        self.with_neg_context = neg_context
        self.with_cos_loss = cos_loss
        # self.with_dist_loss = not no_dist_loss
        self.batch_size = batch_size

        # Pre-trained model settings
        self.model_ver = model_ver
        self.layer = layer
        self.max_seq_len = max_seq_len
        self.embedding_dim = 768
        self.layer_lim = 7
        # self.layer_lim = int(np.ceil(self.premodel.layer_num / 2))  # Upper-half
        if 'large' in self.model_ver:
            self.embedding_dim = 1024
            self.layer_lim = 13

        dropout_rate = 0.1
        self.xl = False
        if model_ver in ['xlm-roberta-base', 'xlm-roberta-large', 'sentence-transformers',
                         'bert-base-multilingual-cased']:
            self.xl = True

        # for disentangle context vector
        self.context_encoder = Encoder(self.embedding_dim, nhead, dropout_rate, k=ff_dim_k)
        # for disentangle sense vector
        self.sense_encoder = Encoder(self.embedding_dim, nhead, dropout_rate, k=ff_dim_k)

    def set_train_corpus(self, dataset, corpus_size, train_size):
        # preparation for batching
        sent_ids = torch.tensor([i for i in range(corpus_size)], dtype=torch.int)
        sent_id_td = TensorDataset(sent_ids)
        self.train_idxs, self.val_idxs = random_split(sent_id_td, [train_size, corpus_size - train_size])

        if type(dataset[0]).__module__ == np.__name__:
            self.pre_encoded = True
            self.bert_src_vecs = dataset[0]
            self.bert_tgt_vecs = dataset[1]
            if self.with_neg_sense:
                self.bert_neg_vecs = dataset[2]
        else:
            self.pre_encoded = False
            self.premodel = Pretrained_Model(self.model_ver, self.layer, max_length=self.max_seq_len)

            targets, sents, positive_targets, positive_sents, neg_sense_targets, neg_sense_sents, neg_context_targets, neg_context_sents = dataset
            self.targets = targets
            self.sents = sents
            self.positive_targets = positive_targets
            self.positive_sents = positive_sents
            self.neg_sense_targets = neg_sense_targets
            self.neg_sense_sents = neg_sense_sents
            self.neg_context_targets = neg_context_targets
            self.neg_context_sents = neg_context_sents

    def train_dataloader(self):
        return DataLoader(self.train_idxs, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_idxs, batch_size=self.batch_size, shuffle=False)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        ### Generate sense representation
        sense = self.sense_encoder(x)
        ### Generate context representation
        context = self.context_encoder(x)

        return sense, context

    def _disentangle(self, x):
        return self.forward(x)

    def _reconstruct(self, sense, context):
        # recon = self.decoder(torch.cat((sense, context), 1))
        recon = (sense + context) / 2
        return recon

    def step(self, batch, batch_idx):
        src_vecs, tgt_vecs, neg_sense_vecs, neg_context_vecs = self.encode_by_bert(batch)

        # Src
        src_y = torch.mean(src_vecs, dim=1)
        src_sense, src_context = self._disentangle(src_vecs)

        # Tgt
        tgt_y = torch.mean(tgt_vecs, dim=1)
        tgt_sense, tgt_context = self._disentangle(tgt_vecs)

        # Negative-sense
        if self.with_neg_sense:
            neg_sense_y = torch.mean(neg_sense_vecs, dim=1)
            neg_sense_sense, neg_sense_context = self._disentangle(neg_sense_vecs)

        # Reconstruction loss
        src_recon = self._reconstruct(src_sense, src_context)
        tgt_recon = self._reconstruct(tgt_sense, tgt_context)
        recon_loss = F.mse_loss(src_recon, src_y) + F.mse_loss(tgt_recon, tgt_y)
        if self.with_neg_sense:
            neg_sense_recon = self._reconstruct(neg_sense_sense, neg_sense_context)
            recon_loss += F.mse_loss(neg_sense_recon, neg_sense_y)

        # Reconstruction: Cross
        ## Combination with tgt
        st_cross_recon = self._reconstruct(src_sense, tgt_context)  # => tgt
        ts_cross_recon = self._reconstruct(tgt_sense, src_context)  # => src
        cross_recon_loss = F.mse_loss(st_cross_recon, tgt_y) + F.mse_loss(ts_cross_recon, src_y)
        if self.with_neg_sense:
            if self.xl:  # Cross-lingual reconstruction
                ## Combination with negative-sense
                sns_cross_recon = self._reconstruct(src_sense, neg_sense_context)  # => tgt
                nst_cross_recon = self._reconstruct(neg_sense_sense, tgt_context)  # => neg_sense
                cross_recon_loss += F.mse_loss(sns_cross_recon, tgt_y) + F.mse_loss(nst_cross_recon, neg_sense_y)
            else:  # Monolingual
                ## Combination with negative-sense
                sns_cross_recon = self._reconstruct(src_sense, neg_sense_context)  # => src
                nss_cross_recon = self._reconstruct(neg_sense_sense, src_context)  # => neg_sense
                cross_recon_loss += F.mse_loss(sns_cross_recon, src_y) + F.mse_loss(nss_cross_recon, neg_sense_y)

        loss = recon_loss + cross_recon_loss
        logs = {
            "recon_loss": recon_loss,
            "cross_recon_loss": cross_recon_loss,
            "loss": loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # AdamW
        optimizer = transformers.AdamW(
            self.parameters(),
            lr=self.lr
        )

        if self.warmup > 0:
            optimizers = [optimizer]
            # Warm-up scheduler
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup)
            schedulers = [{'scheduler': scheduler, 'interval': 'step'}]
            return optimizers, schedulers
        else:
            return optimizer

    def encode_by_bert(self, batch):
        indexes = batch[0]
        neg_sense_vecs, neg_context_vecs = None, None

        if self.pre_encoded:
            src_vecs = torch.stack([torch.from_numpy(self.bert_src_vecs[i]) for i in indexes]).to(self.device)
            tgt_vecs = torch.stack([torch.from_numpy(self.bert_tgt_vecs[i]) for i in indexes]).to(self.device)
            if self.with_neg_sense:
                neg_sense_vecs = torch.stack([torch.from_numpy(self.bert_neg_vecs[i]) for i in indexes]).to(self.device)

        else:
            src_vecs = self.premodel.encode([self.targets[i] for i in indexes], [[self.sents[i]] for i in indexes],
                                            match_all=False, mean=False)

            tgt_vecs = self.premodel.encode([self.positive_targets[i] for i in indexes],
                                            [[self.positive_sents[i]] for i in indexes], match_all=False, mean=False)

            if self.with_neg_sense:
                neg_sense_vecs = self.premodel.encode([self.neg_sense_targets[i] for i in indexes],
                                                      [[self.neg_sense_sents[i]] for i in indexes], match_all=False,
                                                      mean=False)

            # Remove Nan
            src_no_nan_indices = set([i for i in range(len(indexes)) if not torch.any(torch.isnan(src_vecs[i]))])
            tgt_no_nan_indices = set([i for i in range(len(indexes)) if not torch.any(torch.isnan(tgt_vecs[i]))])
            no_nan_indices = src_no_nan_indices & tgt_no_nan_indices
            if self.with_neg_sense:
                negs_no_nan_indices = set(
                    [i for i in range(len(indexes)) if not torch.any(torch.isnan(neg_sense_vecs[i]))])
                no_nan_indices = no_nan_indices & negs_no_nan_indices

            src_vecs = torch.stack([src_vecs[i][self.layer_lim:, :] for i in no_nan_indices]).to(self.device)
            tgt_vecs = torch.stack([tgt_vecs[i][self.layer_lim:, :] for i in no_nan_indices]).to(self.device)
            if self.with_neg_sense:
                neg_sense_vecs = torch.stack([neg_sense_vecs[i][self.layer_lim:, :] for i in no_nan_indices]).to(
                    self.device)

        return src_vecs, tgt_vecs, neg_sense_vecs, neg_context_vecs
