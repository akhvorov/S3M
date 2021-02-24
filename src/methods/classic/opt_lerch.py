import numpy as np
from collections import Counter
from typing import List, Tuple

from methods.base import SimStackModel
from preprocess.seq_coder import SeqCoder


class OptLerchModel(SimStackModel):
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.word2idx = {}
        self.idf = []
        self.N = None
        self.tfidf_cache = {}
        self.words_cache = {}

    def words_set(self, stack_id):
        if stack_id not in self.words_cache:
            words = self.coder(stack_id)
            self.words_cache[stack_id] = set(words)
        return self.words_cache[stack_id]

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'OptLerchModel':
        if self.N is not None:
            raise ValueError("Model already fitted")
        self.coder.fit(unsup_data)
        self.N = len(unsup_data)
        doc_freq = []
        for id in unsup_data:
            for token_id in self.words_set(id):
                if token_id not in self.word2idx:
                    self.word2idx[token_id] = len(self.word2idx)
                    doc_freq.append(0)
                doc_freq[self.word2idx[token_id]] += 1
        for i, v in enumerate(doc_freq):
            self.idf.append(1 + np.log(self.N / v))  # v + 1

        return self

    def transform(self, stack_id):
        if stack_id not in self.tfidf_cache:
            vec = {}
            words = self.coder(stack_id)
            words_freqs = Counter(words)
            for word, freq in words_freqs.items():
                if word not in self.word2idx:
                    idf = np.log(self.N)
                else:
                    idf = self.idf[self.word2idx[word]]
                tf = np.sqrt(words_freqs[word])
                vec[word] = tf * idf ** 2
            self.tfidf_cache[stack_id] = vec
        return self.tfidf_cache[stack_id]

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        anchor_tfidf = self.transform(anchor_id)
        stacks = [self.words_set(stack_id) for stack_id in stack_ids]

        scores = []
        for stack in stacks:
            score = 0
            for word in stack:
                if word not in anchor_tfidf:
                    continue
                tf_idf_pow2 = anchor_tfidf[word]
                score += tf_idf_pow2
            scores.append(score)

        return scores

    def name(self) -> str:
        return self.coder.name() + "_opt_lerch"
