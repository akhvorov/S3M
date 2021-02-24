from abc import ABC, abstractmethod
from typing import List

from data.objects import Stack


class Entry2Seq(ABC):
    @abstractmethod
    def __call__(self, stack: Stack):
        pass

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def remove_equals(words: List[str]) -> List[str]:
    res = []
    for i, w in enumerate(words):
        if (i == 0 or words[i - 1] != w) and w.strip() != '':
            res.append(w)
    return res


class Stack2Seq(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int = 0, sep: str = '.'):
        self.sep = sep
        self.cased = cased
        self.trim_len = trim_len
        self._name = ("" if cased else "un") + "cs" + (f"_tr{trim_len}" if trim_len > 0 else "")

    def __call__(self, stack: Stack) -> List[str]:
        seq = [str(s) for s in stack.frames[::-1]]
        if self.trim_len > 0:
            seq = [self.sep.join(s.split(self.sep)[:-self.trim_len]) for s in seq]
        seq = [s if self.cased else s.lower() for s in seq]
        return seq

    def name(self) -> str:
        return self._name
