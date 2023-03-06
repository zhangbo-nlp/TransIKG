#!/usr/bin/env python3

import json
import os
import re
import sys
from collections import OrderedDict
from operator import itemgetter
from typing import Optional, Tuple

import spacy
from tqdm import tqdm

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.metrics import F1Metric, BleuMetric, RougeMetric
from .build import build


_PUNCS_RE = re.compile(r'[^\w\s]')

_PLOT = 0
_REVIEW = 1
_COMMENTS = 2
_FACT_TABLE = 3
LABEL_ID2STR = {
    _PLOT: 'plot',
    _REVIEW: 'review',
    _COMMENTS: 'comments',
    _FACT_TABLE: 'fact_table'
}

def _path(opt):
    # build the data if it does not exist
    build(opt)

    return os.path.join(opt['datapath'], 'holl_e')


def _remove_duplicate(a_list):
    return list(OrderedDict.fromkeys(a_list))


def _f1_score(true_set, pred_set, eps=sys.float_info.epsilon):
    precision = len(true_set.intersection(pred_set)) / (float(len(pred_set)) + eps)
    recall = len(true_set.intersection(pred_set)) / (float(len(true_set)) + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return f1_score


def _check_continuity(bool_list):
    """Check if all matches are adjoint"""
    matched_indices = [idx for idx, is_match in enumerate(bool_list) if is_match]
    return all(a + 1 == b for a, b in zip(matched_indices[:-1], matched_indices[1:])), matched_indices


class HollETeacher(FixedDialogTeacher):
    """
    Sequence of utterances and responses with background knowledge about movies.

    From the Holl-E dataset. More information found at
    https://github.com/nikitacs16/Holl-E.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('Holl-E Knowledge arguments')
        group.add_argument(
            '--include-knowledge',
            type='bool',
            default=True,
            help='Whether to include the knowledge available to' ' the wizard',
        )
        group.add_argument(
            '--include-checked-sentence',
            type='bool',
            default=True,
            help='Whether to include the Wizard\'s' 'checked sentence',
        )
        group.add_argument(
            '--include-knowledge-separator',
            type='bool',
            default=True,
            help='include special __knowledge__ token between ' 'title and passage',
        )
        group.add_argument(
            '--chosen-topic-delimiter',
            type=str,
            default='\n',
            help='delimiter used when including chosen topic',
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.holle_path = _path(opt)
        self.datatype = opt['datatype'].split(':')[0]
        self.id = 'holl_e'
        self._sent_tok = spacy.load('en_core_web_sm')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', True)
        self.knowledge_separator = opt.get('include_knowledge_separator', True)
        if shared is not None:
            self.episodes = shared['episodes']
        else:
            self.episodes = self.setup_data(self.holle_path)
        self.reset()

    def setup_data(self, path):
        # use test json if valid is given
        json_dtype = self.datatype if not self.datatype.startswith('valid') else 'test'
        episodes_fname = os.path.join(path, f'{json_dtype}_episodes.json')
        if os.path.exists(episodes_fname):
            print('loading: ' + episodes_fname)
            with PathManager.open(episodes_fname) as f:
                episodes = []
                for line in f:
                    episodes.append(json.loads(line))
            return episodes

        # Load raw dataset
        raw_fname = os.path.join(path, f'{json_dtype}_data.json')
        with PathManager.open(raw_fname) as fp:
            episodes = json.load(fp)

        multi_fname = os.path.join(path, 'multi_reference_test.json')
        with PathManager.open(multi_fname) as fp:
            multi_responses = json.load(fp)
        episodes = self._to_wow_format(episodes, multi_responses, json_dtype)
        if self.holle_path:
            episodes_fname = os.path.join(self.holle_path, f'{json_dtype}_episodes.json')
            print(f"Cache preprocessed dataset to {episodes_fname}")
            with PathManager.open(episodes_fname, 'w') as fp:
                for episode in episodes:
                    fp.write(json.dumps(episode) + '\n')

        return episodes

    def _to_wow_format(self, raw_episodes, multi_responses, mode):
        print(f"Convert holle {mode} dataset to wow format")
        episodes = []
        for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
            episode = []
            multi_cnt = 0
            for example_idx in range(0, len(raw_episode['chat']), 2):
                if example_idx + 1 < len(raw_episode['chat']):
                    chosen_topic = raw_episode['movie_name']
                    response = raw_episode['chat'][example_idx + 1]
                    knowledge_sentences = self._get_knowledge_sentences(
                        raw_episode,
                        episode_idx,
                        example_idx,
                        mode,
                    )
                    checked_sentence = knowledge_sentences[0]
                    title = 'no_passages_used' if checked_sentence == 'no_passages_used' else chosen_topic
                    token_knowledge = ' __knowledge__ ' if self.knowledge_separator else ' '
                    formatted_knowledge = '\n'.join([
                        chosen_topic + token_knowledge + k
                        if k != 'no_passages_used'
                        else f'no_passages_used{token_knowledge}no_passages_used'
                        for k in knowledge_sentences
                    ])

                    example = {
                        'id': self.id,
                        'text': raw_episode['chat'][example_idx],
                        'labels': [response],
                        'chosen_topic': chosen_topic,
                        'label_candidates': [],
                    }
                    if self.include_knowledge:
                        example['knowledge'] = formatted_knowledge
                    if self.include_checked_sentence:
                        example['title'] = title
                        example['checked_sentence'] = checked_sentence

                    if mode == 'test':
                        # add multiple responses
                        example['multi_eval_labels'] = [response]
                        example['multi_checked_sentences'] = [checked_sentence]
                        if multi_cnt < len(raw_episode['chat']) // 2:
                            if f'ts_{episode_idx}_{multi_cnt}' in multi_responses.keys():
                                multi_response_id = f'ts_{episode_idx}_{multi_cnt}'
                                for multi_idx in range(len(multi_responses[multi_response_id]['responses'])):
                                    raw_multi_response = multi_responses[multi_response_id]['responses'][multi_idx]
                                    raw_multi_span = multi_responses[multi_response_id]['spans'][multi_idx]
                                    if raw_multi_response != response:
                                        multi_response = _PUNCS_RE.sub('', str(raw_multi_response))
                                        multi_span = _PUNCS_RE.sub('', str(raw_multi_span))
                                        multi_knowledge_sentences = [_PUNCS_RE.sub('', str(x)) for x in knowledge_sentences]
                                        multi_knowledge_idx = self._get_best_match_idx(multi_span, multi_knowledge_sentences, multi_response)
                                        example['multi_eval_labels'].append(raw_multi_response)
                                        example['multi_checked_sentences'].append(knowledge_sentences[multi_knowledge_idx])
                                multi_cnt += 1
                    episode.append(example)
            episodes.append(episode)
        return episodes

    def _get_knowledge_sentences(self, raw_episode, episode_idx, example_idx, mode):
        # Handle special case
        if episode_idx == 5958 and mode == 'train':
            if example_idx in [0, 2]:
                return ['no_passages_used', 'Transformers: Aget of Extinction', '1']
            elif example_idx == 4 or example_idx == 8: # review
                return ['1', 'Transformers: Age of Extinction']
            elif example_idx == 6:
                return ['Transformers: Age of Extinction', '1']

        # Make GT and candidates
        knowledge_candidates = self._get_knowledge_candidates(raw_episode, example_idx)
        gt_knowledge, knowledge_candidates = self._get_gt_knowledge(
            raw_episode, knowledge_candidates, example_idx
        )
        for key, value in knowledge_candidates.items():
            knowledge_candidates[key] = _remove_duplicate(value)

        # Concat GT and candidates
        all_knowledge_sentences = [gt_knowledge]
        for candidates in knowledge_candidates.values():
            all_knowledge_sentences += candidates

        return all_knowledge_sentences

    def _get_knowledge_candidates(self, raw_episode, example_idx):
        label = raw_episode['labels'][example_idx + 1]
        doc = raw_episode['documents']

        plot = self.validate_spacy_sentences(self._sent_tok(doc['plot']))
        review = self.validate_spacy_sentences(self._sent_tok(doc['review']))
        comments = doc['comments']
        fact_table = self._extract_fact_table(doc['fact_table'])
        knowledge_candidates = {
            'plot': plot,
            'review': review,
            'comments': comments,
            'fact_table': fact_table
        }

        return knowledge_candidates

    def _get_gt_knowledge(self, raw_episode, knowledge_candidates, example_idx):
        label = raw_episode['labels'][example_idx + 1]
        label_str = LABEL_ID2STR.get(label, 'none')
        raw_gt_span = raw_episode['spans'][example_idx + 1]
        gt_span = _PUNCS_RE.sub('', raw_gt_span)
        raw_response = raw_episode['chat'][example_idx + 1]
        response = _PUNCS_RE.sub('', raw_response)

        # Find GT knowledge sentence
        if label_str == 'none':
            gt_knowledge = 'no_passages_used'
            gt_knowledge_idx = -1
        else:
            raw_label_candidates = knowledge_candidates[label_str]
            if label_str not in ['plot', 'review']:
                raw_label_candidates = _remove_duplicate(raw_label_candidates)
            label_candidates = [_PUNCS_RE.sub('', x) for x in raw_label_candidates]
            is_gt_in_cand = [gt_span in x for x in label_candidates]
            is_cand_in_gt = [x in gt_span for x in label_candidates]

            num_gt_in_cand = sum(is_gt_in_cand)
            num_cand_in_gt = sum(is_cand_in_gt)

            # Find matched candidate index
            if num_gt_in_cand == 1:  # Exact match
                gt_knowledge_idx = is_gt_in_cand.index(True)
            elif num_gt_in_cand > 1 or label in [_COMMENTS, _FACT_TABLE] or num_cand_in_gt == 0:
                # Find best match
                gt_knowledge_idx = self._get_best_match_idx(gt_span, label_candidates, response)
            elif num_cand_in_gt == 1:  # Inverse exact match
                gt_knowledge_idx = is_cand_in_gt.index(True)
            else:  # Span can exist over multiple sentences
                is_continue, matched_indices = _check_continuity(is_cand_in_gt)
                matched_words = ' '.join([label_candidates[idx] for idx in matched_indices])

                if is_continue and len(gt_span) > len(matched_words):
                    add_front = gt_span.split()[-1] == matched_words.split()[-1]
                    add_rear = gt_span.split()[0] == matched_words.split()[0]
                    index_to_add_front = [] if matched_indices[0] == 0 else [matched_indices[0] - 1]
                    if matched_indices[-1] + 1 == len(label_candidates):
                        index_to_add_rear = []
                    else:
                        index_to_add_rear = [matched_indices[-1] + 1]

                    if add_front:
                        matched_indices = index_to_add_front + matched_indices
                    elif add_rear:
                        matched_indices = matched_indices + index_to_add_rear
                    else:  # Add front & rear
                        matched_indices = index_to_add_front + matched_indices + \
                            index_to_add_rear
                    gt_knowledge_idx = matched_indices
                elif is_continue:
                    gt_knowledge_idx = matched_indices
                else:
                    gt_knowledge_idx = self._get_best_match_idx(
                        gt_span, label_candidates, response)

            # Get GT knowledge
            if isinstance(gt_knowledge_idx, int):
                gt_knowledge = raw_label_candidates[gt_knowledge_idx]
                gt_knowledge_idx = [gt_knowledge_idx]
            elif isinstance(gt_knowledge_idx, list):
                gt_knowledge = ' '.join(itemgetter(*gt_knowledge_idx)(raw_label_candidates))
            else:
                raise ValueError()

            # Remove GT from candidates
            for idx in sorted(gt_knowledge_idx, reverse=True):
                del raw_label_candidates[idx]
            knowledge_candidates[label_str] = raw_label_candidates

        return gt_knowledge, knowledge_candidates

    def _extract_fact_table(self, fact_table):
        if len(fact_table.keys()) == 2:
            return []

        awards = self.validate_sentences(fact_table['awards'])
        taglines = self.validate_sentences(fact_table['taglines'])
        similar_movies = self.validate_sentences(fact_table['similar_movies'])
        box_office = fact_table['box_office']
        if isinstance(box_office, str):
            box_office = [box_office if len(box_office) > 0 else []]
        else:
            box_office = []

        return awards + taglines + similar_movies + box_office

    def _get_best_match_idx(self, gt_span, label_candidates, response):
        gt_span_words = set(gt_span.split())
        response_words = set(response.split())
        label_words_candidates = [
            set(x.split()) for x in label_candidates
        ]

        f1_scores = []
        for label_words_candidate in label_words_candidates:
            f1_scores.append(_f1_score(gt_span_words, label_words_candidate))

        if sum(f1_scores) == 0.0:
            f1_scores = []
            for label_words_candidate in label_words_candidates:
                f1_scores.append(_f1_score(response_words, label_words_candidate))

        max_idx = f1_scores.index(max(f1_scores))

        return max_idx

    def validate_spacy_sentences(self, spacy_sentences):
        def _validate_sent(sent):
            if len(_PUNCS_RE.sub('', sent.text).strip()) > 1:
                return True
            else:
                False

        return [sent.text for sent in spacy_sentences.sents if _validate_sent(sent)]

    def validate_sentences(self, sentences):
        return [sent for sent in sentences if len(sent) > 0]

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def num_examples(self):
        if hasattr(self, '_num_examples_cache'):
            return self._num_examples_cache
        self._num_examples_cache = sum(len(episode) for episode in self.episodes)
        return self._num_examples_cache

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        episode = self.episodes[episode_idx]
        episode_done = entry_idx == (len(episode) - 1)
        entry = episode[entry_idx]

        entry['episode_done'] = episode_done

        return Message(entry)

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ):
        """
        Custom Evaluations for Wizard of Wikipedia.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels, if there were any.
        :param model_response:
            The raw response from the model. Generally you want to rely on the
            text field, but others may be necessary in specific situations.
        """
        if (
            'text' in model_response
            and 'checked_sentence' in teacher_action
            and model_response['text'] is not None
        ):
            self.metrics.add(
                'knowledge_f1',
                F1Metric.compute(model_response['text'], [teacher_action['checked_sentence']]),
            )
            self.metrics.add(
                'multi_knowledge_f1',
                F1Metric.compute(model_response['text'], teacher_action['multi_checked_sentences']),
            )

            self.metrics.add(
                'multi_f1',
                F1Metric.compute(model_response['text'], teacher_action['multi_eval_labels']),
            )

            for k in range(1, 5):  # 1..4
                self.metrics.add(
                    f'multi_bleu-{k}',
                    BleuMetric.compute(model_response['text'], teacher_action['multi_eval_labels'], k),
                )

            r1, r2, rL = RougeMetric.compute_many(
                model_response['text'], teacher_action['multi_eval_labels']
            )
            if r1:
                self.metrics.add('multi_rouge_1', r1)
            if r2:
                self.metrics.add('multi_rouge_2', r2)
            if  rL:
                self.metrics.add('multi_rouge_L', rL)


class DefaultTeacher(HollETeacher):
    pass
