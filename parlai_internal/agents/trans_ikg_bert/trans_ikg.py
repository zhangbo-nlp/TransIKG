#!/usr/bin/env python3

from itertools import chain
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.torch import padded_tensor, neginf
from parlai.core.torch_agent import Batch
from parlai.utils.misc import round_sigfigs

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE

from .modules import TransIKGModel
from .bert_dictionary import BertDictionaryAgent


TOKEN_DIALOG = '__dialog__'


DEFAULT_OPTS = {
    # "learningrate": 5e-4,
    # "optimizer": "adam",
    # "lr_scheduler": "invsqrt",
    # "warmup_updates": 5000,
    # "clip_norm": 0.1,
    # "ffn_size": 512,
    # "embedding_size": 256,
    # "n_heads": 4,
    # "dropout": 0.2,
    # "n_layers": 6,
    # "betas": "0.9,0.98",
    "truncate": 128,
    "add_token_knowledge": True,
    "dict_textfields": "text,labels,chosen_topic,checked_sentence,knowledge,title",
}


class LabelSmoothingNLLoss(nn.Module):
    def __init__(self, epsilon: float = 0.1, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

    def forward(self, log_preds, target):
        n = log_preds.size()[-1]
        # log_preds = F.log_softmax(log_preds, dim=-1)
        log_preds_sum = -log_preds.sum(dim=-1)
        log_preds_sum[target==self.ignore_index] = 0.
        loss = self.reduce_loss(log_preds_sum)
        nll = F.nll_loss(log_preds, target, ignore_index=self.ignore_index,reduction=self.reduction)
        return self.linear_combination(loss / n, nll)


class _GenericWizardAgent(TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.set_defaults(**DEFAULT_OPTS)
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def batchify(self, obs_batch):
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]

        checked_sentences = []
        for obs in reordered_observations:
            checked_sentence = '{} {} {}'.format(
                obs.get('title', ''), TOKEN_KNOWLEDGE, obs.get('checked_sentence', '')
            )
            checked_sentences.append(checked_sentence)

        batch['checked_sentence'] = checked_sentences

        return batch


class TransIkgAgent(_GenericWizardAgent):
    def __init__(self, opt, shared=None):
        self.auxiliary_loss = opt.get('auxiliary_loss', False)
        self.label_smoothing = opt.get('label_smoothing', False)
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2 ** 20))(self._vectorize_text)

        # knowledge truncate defaults to the same as --truncate
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']

        self.history.delimiter_tok = [self.dict[self.history.delimiter]]
        self.use_dialogue_position = opt['use_dialogue_position']

    def _dummy_batch(self, bsz, maxlen):
        batch = super()._dummy_batch(bsz, maxlen)
        batch['know_vec'] = torch.zeros(bsz, 2, 2).long().cuda()
        # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
        ck_mask = (torch.ones(bsz, 2, dtype=torch.uint8) != 0).cuda()
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = torch.zeros(bsz).long().cuda()
        batch['use_cs_ids'] = True
        batch['text_segments'] = None
        return batch

    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.
        If you need to add additional tokens to the dictionary, this is likely the right
        place to do it.
        """
        d = super().build_dictionary()
        d[self.opt['delimiter']] = 999999999
        return d

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def build_criterion(self):
        # set up criteria
        if self.label_smoothing:
            return LabelSmoothingNLLoss(epsilon=0.1, ignore_index=self.NULL_IDX, reduction='none')
        else:
            return nn.NLLLoss(ignore_index=self.NULL_IDX, reduction='none')

    def compute_loss(self, batch, return_output=False):
        # first compute our regular forced decoding loss
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        if not self.auxiliary_loss or self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            notnull = batch.label_vec.ne(self.NULL_IDX)
            num_tokens = notnull.long().sum().item()
            encoder_states = model_output[2]
            ctx_know_attn = encoder_states[-1]

            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc

            know_loss = torch.nn.functional.cross_entropy(
                ctx_know_attn, batch.cs_ids, reduction='mean'
            )
            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            # in the original paper the loss was scaled by num_tokens for both
            # know_loss and token_loss
            # know_loss /= num_tokens
            # loss = (
            #     1 - self.knowledge_alpha
            # ) * token_loss + self.knowledge_alpha * know_loss
            loss = token_loss + self.knowledge_alpha * know_loss

        if return_output:
            return loss, model_output
        else:
            return loss

    def reset_metrics(self):
        super().reset_metrics()
        if self.auxiliary_loss:
            self.metrics['bsz'] = 0.0
            self.metrics['know_acc'] = 0.0
            self.metrics['know_loss'] = 0.0
            self.metrics['know_chance'] = 0.0

    def report(self):
        r = super().report()
        if self.auxiliary_loss:
            bsz = max(self.metrics['bsz'], 1)
            for k in ['know_loss', 'know_acc', 'know_chance']:
                # round and average across all items since last report
                r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
        return r

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            # make a copy of the list to prevent the future padding step from
            # being destructive
            return list(obs['knowledge_parsed'])

        if 'checked_sentence' not in obs:
            # interactive time. we're totally on our own
            obs_know = [
                k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
            ]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            return obs['knowledge_parsed']

        checked_sentence = '{} {} {}'.format(
            obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence']
        )
        # grab all the nonempty knowledge
        obs_know = [
            k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
        ]
        obs_know = [k for k in obs_know if k]

        # we want the correct knowledge to always be in index 0
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        return obs['knowledge_parsed']

    def batchify(self, obs_batch):
        """
        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]

        # first parse and compile all the knowledge together
        all_knowledges = []  # list-of-lists knowledge items for each observation
        knowledge_counts = []  # how much knowledge each observation gets
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            # downsample if desired
            if (
                is_training
                and self.max_knowledge
                and len(obs_know) > self.max_knowledge
            ):
                # offset by one so that we don't choose 0
                keepers = 1 + np.random.choice(
                    len(obs_know) - 1, self.max_knowledge, False
                )
                # correct answer is always the first one
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))

        # now we want to actually pack this into a tensor, along with the mask
        N = len(reordered_observations)
        K = max(knowledge_counts)
        # round out the array so everything is equally sized
        for i in range(N):
            all_knowledges[i] += [''] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))

        knowledge_vec = [
            self._vectorize_text(
                # the beginning of the sentence is more useful
                k,
                truncate=self.knowledge_truncate,
                add_end=True,
                truncate_left=False,
            )
            for k in flattened_knowledge
        ]
        knowledge_vec, _ = padded_tensor(
            knowledge_vec, pad_idx=self.NULL_IDX, left_padded=True
        )
        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)

        # knowledge mask is a N x K tensor saying which items we're allowed to
        # attend over
        bsz = len(reordered_observations)
        ck_mask = torch.zeros(bsz, K, dtype=torch.uint8)
        for i, klen in enumerate(knowledge_counts):
            ck_mask[i, :klen] = 1
        ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
        # and the correct labels
        cs_ids = torch.LongTensor(bsz).zero_()

        if self.use_cuda:
            if batch.text_vec is not None:
                dev = batch.text_vec.device
            else:
                assert batch.label_vec is not None, "need label_vec for _generate"
                dev = batch.label_vec.device
            knowledge_vec = knowledge_vec.to(dev)
            ck_mask = ck_mask.to(dev)
            cs_ids = cs_ids.to(dev)

        text_segments = torch.zeros_like(batch.text_vec)
        if self.use_dialogue_position:
            text_split_ids = (batch.text_vec == self.history.delimiter_tok[0]).nonzero().tolist()
            text_split_ids.reverse()
            for i, j in text_split_ids:
                text_segments[i, :j + 1] += 1

        batch['know_vec'] = knowledge_vec
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = cs_ids
        batch['use_cs_ids'] = is_training
        batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
        batch['text_segments'] = text_segments

        return batch

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group("EndToEnd Agent")
        group.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        group.add_argument(
            '--knowledge-truncate',
            type=int,
            default=32,
            help='Knowledge truncation field. Defaults to same as --truncate.',
        )
        group.add_argument(
            '--max-knowledge',
            type=int,
            help='Reduce the amount of negative knowledge at train time.',
        )
        group.add_argument('--auxiliary-loss', type='bool', default=False)
        group.add_argument('--fusion-attn', type=str, default='dot')
        group.add_argument('--use-dialogue-position', type='bool', default=False)
        group.add_argument('--use-correlation-integration', type='bool', default=False)
        group.add_argument('--use-overall-integration', type='bool', default=False)
        group.add_argument('--use-copy', type='bool', default=False)
        group.add_argument('--label-smoothing', type='bool', default=False)
        group.set_defaults(dict_maxexs=0)  # skip building dictionary

        return parser

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            beams = [
                self._treesearch_factory(dev)
                .set_context(self._get_context(batch, batch_idx))
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, _, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score, encoder_states, incr_state)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts].unsqueeze(-1).repeat(1, beam_size)
                prefix_score = score.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.NULL_IDX)
                score[prefix_mask] = neginf(score.dtype)
                score[prefix_mask] = score[prefix_mask].scatter_(
                    -1,
                    prefix_toks[prefix_mask].unsqueeze(-1),
                    prefix_score[prefix_mask],
                )
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.know_vec,
            batch.ck_mask,
            batch.text_segments,
        )

    def build_model(self):
        self.model = TransIKGModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.word_embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model = self.model.cuda()
        return self.model
