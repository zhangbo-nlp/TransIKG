#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent

try:
    from transformers import BertTokenizer
except ImportError:
    raise ImportError(
        'BERT needs transformers installed. \n '
        'pip install transformers'
    )


SPECIAL_TOKENS = {"bos_token": "[unused0]", "eos_token": "[unused1]"}


class BertDictionaryAgent(HuggingFaceDictionaryAgent):
    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    @property
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """
        return False

    @property
    def skip_decode_special_tokens(self) -> bool:
        """
        Whether to skip special tokens when converting tokens to text.
        """
        return True

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def override_special_tokens(self, opt):
        self.hf_tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.start_idx = self.hf_tokenizer.bos_token_id
        self.end_idx = self.hf_tokenizer.eos_token_id
        self.pad_idx = self.hf_tokenizer.pad_token_id
        self.null_idx = self.hf_tokenizer.pad_token_id

        self.start_token = self.hf_tokenizer.bos_token
        self.end_token = self.hf_tokenizer.eos_token
        self.null_token = self.hf_tokenizer.pad_token
        self.unk_token = self.hf_tokenizer.unk_token

        self._unk_token_idx = self.hf_tokenizer.unk_token_id

        # set tok2ind for special tokens
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.null_token] = self.null_idx
        # set ind2tok for special tokens
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.null_idx] = self.null_token

        self.unk_idx = self[self.unk_token]
