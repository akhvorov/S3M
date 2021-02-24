import pickle
from abc import ABC
from functools import lru_cache
from typing import List, Dict, Tuple, Union

import numpy as np
from collections import Counter

from methods.base import SimStackModel
from preprocess.seq_coder import SeqCoder


class TfIdfComputer:
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.word2idx = {}
        self.doc_freq = []
        self.N = None

    def fit(self, stack_ids: List[int]) -> 'TfIdfComputer':
        if self.N is not None:
            print("TfIdf model already fitted (skipped)")
            return self
        self.coder.fit(stack_ids)
        texts = [" ".join(self.coder.to_seq(id)) for id in stack_ids]
        self.N = len(texts)
        for text in texts:
            for word in set(text.split(' ')):
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.doc_freq.append(0)
                self.doc_freq[self.word2idx[word]] += 1
        for i, v in enumerate(self.doc_freq):
            self.doc_freq[i] = 1 + np.log(self.N / v)  # v + 1
        return self

    def idf(self, frame: str, default: float = 0.):
        if frame not in self.word2idx:
            return default
        return self.doc_freq[self.word2idx[frame]]

    @lru_cache(maxsize=20_000)
    def transform(self, stack_id: int) -> Dict[str, Tuple[float, float]]:
        vec = {}
        words = self.coder.to_seq(stack_id)
        words_freqs = Counter(words)
        for word, freq in words_freqs.items():
            if word not in self.word2idx:
                idf = np.log(self.N)
            else:
                idf = self.doc_freq[self.word2idx[word]]
            tf = np.sqrt(words_freqs[word])
            vec[word] = tf, idf
        return vec


class IntTfIdfComputer:
    def __init__(self, coder: SeqCoder, ns: Tuple[int] = None):
        self.coder = coder
        self.ns = ns or (1,)
        self.word2idx = {}
        self.doc_freq = []
        self.N = None
        self.words_cache = {}

    def fit(self, stack_ids: List[int]) -> 'IntTfIdfComputer':
        if self.N is not None:
            print("TfIdf model already fitted (skipped)")
            return self
        self.coder.fit(stack_ids)
        self.N = len(stack_ids)
        for stack_id in stack_ids:
            for word in self.words_tfs(stack_id).keys():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.doc_freq.append(0)
                self.doc_freq[self.word2idx[word]] += 1
        for i, v in enumerate(self.doc_freq):
            self.doc_freq[i] = 1 + np.log(self.N / v)  # v + 1
        return self

    def words_tfs(self, stack_id) -> Dict[Tuple[int, ...], int]:
        if stack_id not in self.words_cache:
            # words = set(self.coder(stack_id))
            words = self.coder.ngrams(stack_id, ns=self.ns)
            self.words_cache[stack_id] = words
        return self.words_cache[stack_id]

    @lru_cache(maxsize=100_000)
    def transform(self, stack_id: int) -> Dict[int, Tuple[float, float]]:
        vec = {}
        words_freqs = self.words_tfs(stack_id)
        for word, freq in words_freqs.items():
            if word not in self.word2idx:
                idf = np.log(self.N)
            else:
                idf = self.doc_freq[self.word2idx[word]]
            tf = np.sqrt(words_freqs[word])
            vec[word] = tf, idf
        return vec

    def name(self) -> str:
        return self.coder.name() + "_inttfidf"


class TfIdfModule:
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.tfidf_vectorizer = TfIdfComputer(self.coder)

    def fit(self, train_stacks: List[int]):
        self.tfidf_vectorizer.fit(train_stacks)
        return self

    def name(self) -> str:
        return self.coder.name() + "_tfidf"

    def save(self, name: str = ""):
        with open("models/" + self.name() + "_" + name + ".model", 'wb') as f:
            pickle.dump(self, f)

    def load(self, name: str = "") -> Union[None, 'TfIdfModule']:
        path = "models/" + self.name() + "_" + name + ".model"
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(e)
            return None


class TfIdfBaseModel(SimStackModel, ABC):
    def __init__(self, coder: SeqCoder):
        self.tfidf_module = TfIdfModule(coder)

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'TfIdfBaseModel':
        self.tfidf_module.fit(unsup_data)
        return self

    def save(self, name: str = ""):
        self.tfidf_module.save(name)

    def load(self, name: str = "") -> Union[None, 'TfIdfBaseModel']:
        tfidf_module = self.tfidf_module.load(name)
        if tfidf_module is not None:
            print("Load model")
            self.tfidf_module = tfidf_module
        return self
