#!/usr/bin/env python3

from typing import Dict, Tuple, Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper


LAYER_NORM_EPS = 1e-5  # Epsilon for layer norm.


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


class TransIKGModel(TorchGeneratorModel):
    """
    Implements a full generator model, with one encoder and one decoder.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.use_copy = opt['use_copy']

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0,
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024
        n_segments = opt.get('n_segments', 0)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = TransIKGEncoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            n_positions=n_positions,
            n_segments=n_segments,
        )

        self.decoder = TransIKGDecoder(
            opt, self.embeddings, n_positions=n_positions
        )

        if self.use_copy:
            self.context_attn_linear = nn.Linear(opt['embedding_size'], opt['n_heads'])
            self.know_attn_linear = nn.Linear(opt['embedding_size'], opt['n_heads'])
            self.know_gate_control = nn.Linear(opt['embedding_size'], 1)
            self.final_merge = nn.Linear(opt['embedding_size'], 3)
            nn.init.xavier_uniform_(self.context_attn_linear.weight)
            nn.init.xavier_uniform_(self.know_attn_linear.weight)
            nn.init.xavier_uniform_(self.know_gate_control.weight)
            nn.init.xavier_uniform_(self.final_merge.weight)

    def decode_forced(self, encoder_states, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        latent, new_encoder_states, decoder_states = self.decoder(inputs, encoder_states)
        logits = self.output(latent, new_encoder_states, decoder_states)
        _, preds = logits.max(dim=2)
        return logits, preds

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        context_encoded, context_mask, src_tokens, know_encoded, \
        know_mask, know_tokens, fusion_enc, text_segments, ck_attn = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(context_encoded.device)
        context_encoded = torch.index_select(context_encoded, 0, indices)
        context_mask = torch.index_select(context_mask, 0, indices)
        src_tokens = torch.index_select(src_tokens, 0, indices)
        know_encoded = torch.index_select(know_encoded, 0, indices)
        know_mask = torch.index_select(know_mask, 0, indices)
        know_tokens = torch.index_select(know_tokens, 0, indices)
        fusion_enc = torch.index_select(fusion_enc, 0, indices)
        ck_attn = torch.index_select(ck_attn, 0, indices)
        text_segments = torch.index_select(text_segments, 0, indices)

        return context_encoded, context_mask, src_tokens, know_encoded, know_mask,\
               know_tokens, fusion_enc, text_segments, ck_attn

    def reorder_decoder_incremental_state(
            self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor, encoder_states, decoder_states):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)

        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)

        prob = F.softmax(output, dim=-1)
        if self.use_copy:
            _, _, src_tokens, _, _, know_tokens, _, _, ck_attn = encoder_states

            # Select the output of the last decoder layer
            ids = len(decoder_states) - 1
            context_attn = decoder_states[ids]['context_attn']['attn_weights']
            know_attn = decoder_states[ids]['know_attn']['attn_weights']
            context_out = decoder_states[ids]['context_out']
            know_out = decoder_states[ids]['know_out']

            batch_size, n_heads, resp_len, src_len = context_attn.size()
            context_head_weights = F.softmax(self.context_attn_linear(context_out), dim=-1).transpose(1, 2)
            context_attn = torch.sum(context_attn * context_head_weights.unsqueeze(-1), dim=1)

            _, _, resp_len, know_len = know_attn.size()
            know_head_weights = F.softmax(self.know_attn_linear(know_out), dim=-1).transpose(1, 2)
            know_attn = torch.sum(know_attn * know_head_weights.unsqueeze(-1), dim=1)

            merge_weights = F.softmax(self.final_merge(tensor), dim=-1)

            context_prob = context_attn * merge_weights[:, :, 1:2]

            ck_len = ck_attn.size()[-1]
            kt_len = int(know_len / ck_len)
            ck_attn = ck_attn.unsqueeze(-1).expand(batch_size, ck_len, kt_len).reshape(batch_size, know_len)
            know_gate = torch.sigmoid(self.know_gate_control(know_out))
            know_prob = know_attn * know_gate + ck_attn.unsqueeze(1).repeat(1, resp_len, 1) / kt_len * (1 - know_gate)
            know_prob = know_prob * merge_weights[:, :, 2:3]

            prob = prob * merge_weights[:, :, :1]
            prob = prob.scatter_add(-1, src_tokens.unsqueeze(1).repeat(1, tensor.size()[1], 1), context_prob)
            prob = prob.scatter_add(-1, know_tokens.unsqueeze(1).repeat(1, tensor.size()[1], 1), know_prob)

        log_prob = torch.log(prob + 1e-16)

        return log_prob


def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()

    divisor[divisor < 1] = 1
    sentence_sums /= divisor
    return sentence_sums


class TransIKGEncoder(nn.Module):
    def __init__(self, opt,
                 dictionary,
                 embeddings,
                 pad_idx,
                 reduction_type,
                 n_positions,
                 n_segments):
        super().__init__()
        # The transformer takes care of most of the work, but other modules
        # expect us to have an embeddings available
        self.embeddings = embeddings
        self.embed_dim = opt['embedding_size']
        self.hidden_dim = opt['ffn_size']
        self.fusion_attn = opt['fusion_attn']
        self.use_dialogue_position = opt['use_dialogue_position']

        if self.use_dialogue_position:
            text_n_segments = opt['history_size'] if opt['history_size'] >= 1 else opt['max_knowledge']
        else:
            text_n_segments = n_segments

        # if self.use_dialogue_position:
        #     text_n_segments = opt['history_size'] if opt['history_size'] >= 1 else opt['max_knowledge']
        #     self.dialogue_position_embeddings = nn.Embedding(text_n_segments, self.embed_dim)
        #     nn.init.normal_(self.dialogue_position_embeddings.weight, 0, self.embed_dim ** -0.5)

        # self.context_encoder = TransformerEncoder(
        #     opt, dictionary, embeddings, pad_idx, reduction_type, n_positions, n_segments
        # )
        # self.knowledge_encoder = TransformerEncoder(
        #     opt, dictionary, embeddings, pad_idx, reduction_type, n_positions, n_segments
        # )

        self.context_encoder = TransformerEncoder(
            opt, dictionary, embeddings, pad_idx, reduction_type, n_positions, text_n_segments
        )
        self.knowledge_encoder = TransformerEncoder(
            opt, dictionary, embeddings, pad_idx, reduction_type, n_positions, n_segments
        )

        if self.fusion_attn == 'mlp':
            self.linear_context = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
            self.linear_know = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_dim, 1, bias=False)
            nn.init.xavier_uniform_(self.linear_context.weight)
            nn.init.xavier_uniform_(self.linear_know.weight)
            nn.init.xavier_uniform_(self.v.weight)

        self.activation = opt['activation']
        self.relu_dropout = opt.get('relu_dropout', 0.0)

        self.fusion_ffn = TransIntegration(
            self.embed_dim, self.hidden_dim, relu_dropout=self.relu_dropout, activation=self.activation
        )

    def forward(self, src_tokens, know_tokens, ck_mask, text_segments):
        # encode the context, pretty basic
        context_encoded, context_mask = self.context_encoder(src_tokens, segments=text_segments)
        # context_encoded, context_mask = self.context_encoder(src_tokens)
        # if self.use_dialogue_position:
        #     if text_segments is None:
        #         text_segments = torch.zeros_like(src_tokens)  # type: ignore
        #     context_encoded = context_encoded + self.dialogue_position_embeddings(text_segments)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.knowledge_encoder(know_flat)
        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask, sqrt=False)
        know_use = universal_sentence_embedding(know_encoded, know_mask, sqrt=False)

        # remash it back into the shape we need
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)

        # compute the attention weights of knowledge
        if self.fusion_attn == 'mlp':
            hidden = self.linear_context(context_use.unsqueeze(1)) + self.linear_know(know_use)
            ck_attn = self.v(self.tanh(hidden)).squeeze(-1)
        else:
            sqrt_dim = np.sqrt(self.embed_dim)
            ck_attn = torch.bmm(know_use / sqrt_dim, context_use.unsqueeze(-1) / sqrt_dim).squeeze(-1)

        # fill with near -inf
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoded.dtype))
        ck_attn = F.softmax(ck_attn, dim=-1)

        cs_encoded = torch.bmm(ck_attn.unsqueeze(1), know_use)

        # Fusion vector of context and knowledge, shape: (batch, 1, dim)
        fusion_enc = self.fusion_ffn(torch.cat([context_use.unsqueeze(1), cs_encoded], dim=-1))

        know_encoded = know_encoded.reshape(N, -1, self.embed_dim)
        know_mask = know_mask.reshape(N, -1)
        know_tokens = know_tokens.reshape(N, -1)

        return context_encoded, context_mask, src_tokens, know_encoded, \
               know_mask, know_tokens, fusion_enc, text_segments, ck_attn


def create_position_codes(n_pos, dim, out):
    """
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)


def get_n_positions_from_options(opt: Opt):
    """
    Determine n_positions from options dict.
    """
    if opt.get('n_positions'):
        # if the number of positions is explicitly provided, use that
        n_positions = opt['n_positions']
    else:
        # else, use the worst case from truncate
        n_positions = max(
            opt.get('truncate') or 0,
            opt.get('text_truncate') or 0,
            opt.get('label_truncate') or 0,
        )
        if n_positions == 0:
            # default to 1024
            n_positions = 1024
    if n_positions < 0:
        raise ValueError('n_positions must be positive')
    return n_positions


class TransIKGDecoder(nn.Module):
    def __init__(
        self,
        opt: Opt,
        embeddings: Optional[nn.Embedding] = None,
        n_positions: Optional[int] = None,
    ):
        super().__init__()

        def _default(val, default):
            return val if val is not None else default

        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.activation = opt.get('activation', 'relu')
        self.variant = opt.get('variant', 'aiayn')

        self.embeddings_scale = opt.get('embeddings_scale', True)
        dropout_frac = opt.get('dropout', 0.0)
        self.dropout = nn.Dropout(p=dropout_frac)  # --dropout

        self.n_positions = _default(n_positions, get_n_positions_from_options(opt))

        self.use_dialogue_position = opt['use_dialogue_position']

        self.out_dim = self.embedding_size
        assert (
                self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embeddings

        if self.variant == 'bart':
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}, please choose from [aiayn, bart].".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size ** -0.5
            )

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransIKGDecoderLayer(
                    opt,
                    self.n_heads,
                    self.embedding_size,
                    self.ffn_size,
                    attention_dropout=opt.get('attention_dropout', 0.0),
                    relu_dropout=opt.get('relu_dropout', 0.0),
                    dropout=dropout_frac,
                    activation=self.activation,
                    variant=self.variant,
                )
            )

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segements:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embeded input and mask
        """
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        return tensor

    def forward_layers(
            self,
            tensor: torch.Tensor,
            context_out: torch.Tensor,
            context_mask: torch.Tensor,
            know_out: torch.Tensor,
            know_mask: torch.Tensor,
            fusion_out: torch.Tensor,
            text_segments: torch.Tensor,
            incr_state: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param context_out
            context encoder outputs
        :param context_mask:
            context encoder output mask
        :param know_out:
            knowledge encoder outputs
        :param know_mask:
            knowledge encoder output mask
        :param fusion_out:
            Fusion of context and knowledge
        :param incr_state:
            Dict mapping layer_idx to incremental state

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, context_out, context_mask, know_out, know_mask, fusion_out, text_segments, incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    x=tensor,
                    context_out=context_out,
                    context_mask=context_mask,
                    know_out=know_out,
                    know_mask=know_mask,
                    fusion_out=fusion_out,
                    text_segments=text_segments,
                    incr_state=incr_state.get(idx),
                )

        return tensor, new_incr_state

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        context_out, context_mask, context_tokens, know_out, know_mask, \
            know_tokens, fusion_out, text_segments, ck_attn = encoder_state

        seq_len = input.size(1)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input.device
        ).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, context_out, context_mask, know_out, know_mask, fusion_out, text_segments, incr_state
        )

        return tensor, encoder_state, new_incr_state

    def _apply_model_parallel(self, tensor, context_out, context_mask, know_out, know_mask,
                              fusion_out, text_segments, incr_state):
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split(
            (tensor, context_out, context_mask, know_out, know_mask, fusion_out, text_segments, incr_state)
        )
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        new_incr_state = [{} for _ in chunks]

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, s_context_out, s_context_mask, s_know_out, s_know_mask, \
            s_fusion_out, s_text_segments, s_incr_state = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, new_incr_state[chunk_idx][layer_no] = self.layers[layer_no](
                    x=s_tensor,
                    context_out=s_context_out,
                    context_mask=s_context_mask,
                    know_out=s_know_out,
                    know_mask=s_know_mask,
                    fusion_out=s_fusion_out,
                    text_segments=s_text_segments,
                    incr_state=s_incr_state.get(layer_no),
                )
            # don't move incr state, it's always on the correct device
            s_tensor, s_context_out, s_context_mask, s_know_out, s_know_mask, \
                s_fusion_out, s_text_segments = PipelineHelper.chunk_to(
                (s_tensor, s_context_out, s_context_mask, s_know_out, s_know_mask,s_fusion_out, s_text_segments),
                next_device
            )
            chunks[chunk_idx] = (
                s_tensor, s_context_out, s_context_mask, s_know_out,
                s_know_mask, s_fusion_out, s_text_segments, s_incr_state
            )

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: PipelineHelper.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state


def _create_selfattn_mask(x):
    # figure out how many timestamps we need
    bsz = x.size(0)
    time = x.size(1)
    # make sure that we don't look into the future
    mask = torch.tril(x.new(time, time).fill_(1))
    # broadcast across batch
    mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    return mask


class TransIKGDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        opt,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.use_dialogue_position = opt['use_dialogue_position']
        self.use_correlation_integration = opt['use_correlation_integration']
        self.use_overall_integration = opt['use_overall_integration']

        if self.use_dialogue_position:
            text_n_segments = opt['history_size'] if opt['history_size'] >= 1 else opt['max_knowledge']
        else:
            text_n_segments = opt['n_segments']

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.self_norm = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        if self.use_correlation_integration:
            self.ci_ffn = TransIntegration(
                embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation, ci_ffn=True
            )
            self.ci_norm = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.context_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout, n_segments=text_n_segments,
        )
        self.context_norm = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.know_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.know_norm = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        if self.use_overall_integration:
            self.oi_ffn = TransIntegration(
                embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation, oi_ffn=True,
            )
            self.oi_norm = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        else:
            self.simple_integration = nn.Linear(embedding_size * 3, embedding_size)
            nn.init.xavier_uniform_(self.simple_integration.weight)

    def forward(self, x, context_out, context_mask, know_out, know_mask, fusion_out, text_segments, incr_state=None):
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = _create_selfattn_mask(x)

        if self.use_correlation_integration:
            # Correlation Integration
            residual = x
            x = self.ci_ffn(x, fusion_out)
            x = self.dropout(x)
            x = residual + x
            x = self.ci_norm(x)

        # first self attn
        residual = x
        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            use_dialogue_position=False,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = self.self_norm(x)

        residual = x

        # context_encoder_attn_layer_norm
        x_c = x
        x_c, final_context_attn_incr_state = self.context_attention(
            query=x_c,
            key=context_out,
            value=context_out,
            mask=context_mask,
            segments=text_segments,
            incr_state=incr_state.get('context_attn'),
            static_kv=True,
            use_dialogue_position=self.use_dialogue_position,
        )[:2]
        x_c = self.dropout(x_c)  # --dropout
        x_c = residual + x_c
        x_c = self.context_norm(x_c)

        # know_encoder_attn_layer_norm
        x_k = x
        x_k, final_know_attn_incr_state = self.know_attention(
            query=x_k,
            key=know_out,
            value=know_out,
            mask=know_mask,
            incr_state=incr_state.get('konw_attn'),
            static_kv=True,
            use_dialogue_position=False,
        )
        x_k = self.dropout(x_k)  # --dropout
        x_k = residual + x_k
        x_k = self.know_norm(x_k)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'context_attn': final_context_attn_incr_state,
            'know_attn': final_know_attn_incr_state,
            'context_out': x_c,
            'know_out': x_k,
        }

        if self.use_overall_integration:
            # All For One Integration
            residual = x + x_c + x_k
            x_final = self.oi_ffn(x, x_c, x_k)
            x_final = self.dropout(x_final)  # --dropout
            x_final = residual + x_final
            x_final = self.oi_norm(x_final)
        else:
            x_final = self.simple_integration(torch.cat([x, x_c, x_k], dim=-1))

        return x_final, new_incr_state

    def reorder_incremental_state(
            self, incremental_state: Dict[str, dict], inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'context_attn': self.context_attention,
            'know_attn': self.know_attention,
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }


class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(self, n_heads, dim, dropout=0., n_segments=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.n_segments = n_segments

        if n_segments > 0:
            # bias = torch.zeros(n_segments)
            # self.bias = torch.nn.Parameter(bias)
            self.bias_linear = torch.nn.Linear(dim, 1)
            self.hardtanh = torch.nn.Hardtanh(min_val=-0.5, max_val=0.5)

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(  # type: ignore
        # TODO: remove type ignore with pytorch 1.5:
        # https://github.com/pytorch/pytorch/pull/31057
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        segments: Optional[torch.Tensor] = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
        use_dialogue_position: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: tensor in which True means that we are allowing attention and False
          means we are blocking it. Mask is:
          - [B, key_len] (encoder self-attn and decoder enc/dec attn)
          - [B, query_len, key_len] (decoder self-attn)
          - [B, 1, 1] (decoder self-attn with incr_state caching)
        :param segments: segments of input text
        :param incr_state: dictionary with values representing the previous states of
          the key, value, and mask
        :param static_kv: True if the key and value are held constant during decoding
          (as in encoder/decoder attention)
        :param use_dialogue_position: True if using segment attention
        :return: (final attended tensor, new incremental state)
        """

        batch_size, query_len, dim = query.size()
        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                    .contiguous()
                    .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        assert key is not None  # let mypy know we sorted this
        _, _key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))

        # Prepend incremental states. For each of the key, value, and mask, see if
        # a previous incremental state exists, and if so, reshape it to match the shape
        # of the new state. Concatenate the previous and new states to match what the
        # full state would have been if we had not cached. (If we are using static_kv,
        # these three states are unchanging, so just re-use the cached states.)
        if incr_state is None:
            incr_state = {}
        if 'prev_key' in incr_state:
            prev_key = incr_state['prev_key'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, prepare_head(self.k_lin(key))], dim=1)
        else:
            k = prepare_head(self.k_lin(key))
        if 'prev_value' in incr_state:
            prev_value = incr_state['prev_value'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, prepare_head(self.v_lin(value))], dim=1)
        else:
            v = prepare_head(self.v_lin(value))
        if 'prev_mask' in incr_state:
            if static_kv:
                mask = incr_state['prev_mask']
            else:
                mask = torch.cat([incr_state['prev_mask'], mask], dim=2)
                # Prepend along the key_len dimension (analogous to
                # incr_state['prev_key'])

        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        if use_dialogue_position and segments is not None:
            m = self.n_segments
            dpf = (torch.cos(np.pi*segments)  - 4/m * segments + 7 + (3*m+2)/(m+1)) / 8 +\
                 self.hardtanh(self.bias_linear(key)).squeeze(-1)
            dpf = (
                dpf
                    .unsqueeze(1)
                    .repeat(1, n_heads, 1, 1)
                    .view(batch_size * n_heads, full_key_len)
            ).unsqueeze(-1)

            dot_prod = q.div_(scale).bmm((k * dpf).transpose(1, 2))

        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
                .view(batch_size, 1, -1, full_key_len)
                .repeat(1, n_heads, 1, 1)
                .expand(batch_size, n_heads, query_len, full_key_len)
                .view(batch_size * n_heads, query_len, full_key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(
            dot_prod, dim=-1, dtype=torch.float  # type: ignore
        ).type_as(query)

        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
                .view(batch_size, n_heads, query_len, dim_per_head)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, query_len, dim)
        )

        # Save new incremental states. We reshape to allow for reordering along batch
        # dimension.
        new_incr_state = {
            'prev_key': k.view(batch_size, n_heads, -1, dim_per_head),
            'prev_value': v.view(batch_size, n_heads, -1, dim_per_head),
            'prev_mask': mask,
            'attn_weights': attn_weights.view(batch_size, n_heads, query_len, -1),
        }

        out = self.out_lin(attentioned)

        return out, new_incr_state

    def reorder_incremental_state(
            self, incremental_state: Dict[str, torch.Tensor], inds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Reorder the input incremental-state tensors.
        """
        return {
            key: torch.index_select(val, 0, inds.to(val.device)).contiguous()
            for key, val in incremental_state.items()
        }


class TransIntegration(nn.Module):
    """
    Implements the Integration part of the TransIKG.
    """

    def __init__(self, dim, dim_hidden, relu_dropout=0., activation='relu', ci_ffn=False, oi_ffn=False):
        super(TransIntegration, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = F.gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        if ci_ffn:
            self.lin1 = nn.Linear(dim * 2, dim_hidden)
            self.lin_r = nn.Linear(dim * 2, dim)
            nn.init.xavier_uniform_(self.lin_r.weight)
        elif oi_ffn:
            self.lin1 = nn.Linear(dim * 3, dim_hidden)
            self.lin_c = nn.Linear(dim * 2, dim)
            self.lin_k = nn.Linear(dim * 2, dim)
            nn.init.xavier_uniform_(self.lin_c.weight)
            nn.init.xavier_uniform_(self.lin_k.weight)
        else:
            self.lin1 = nn.Linear(dim * 2, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x, y=None, z=None):
        """
        Forward pass.
        """
        if y is not None and z is None:
            y = y.expand(x.size())
            r = torch.sigmoid(self.lin_r(torch.cat([x, y], dim=-1)))
            x = torch.cat([x, r * y], dim=-1)
        if z is not None:
            c = torch.sigmoid(self.lin_c(torch.cat([x, y], dim=-1)))
            k = torch.sigmoid(self.lin_k(torch.cat([x, z], dim=-1)))
            x = torch.cat([x, c * y, k * z], dim=-1)
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x
