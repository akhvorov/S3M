from abc import ABC
from typing import Tuple, List, Iterable
import numpy as np

from data.buckets.issues_data import StackAdditionState


class PairSimSelector(ABC):
    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    def generate(self, events: List[StackAdditionState]) -> Iterable[Tuple[int, int, int]]:
        for event in events:
            st_id = event.st_id
            try:
                st_ids, target = self(event)
                for sid, sim in zip(st_ids, target):
                    yield st_id, sid, sim
            except:
                pass


class RandomPairSimSelector(PairSimSelector):
    def __init__(self, size: int = None):
        self.size = size

    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        good_stacks = np.random.permutation([(sid, 1) for sid in event.issues[event.is_id].stacks])[:self.size]
        bad_issues = np.random.permutation(list(set(event.issues.keys() - {event.is_id})))
        bad_stacks = []
        for iid in bad_issues:
            istacks = [[sid, 0] for sid in event.issues[iid].stacks]
            if not istacks:
                continue
            bad_stacks.append(istacks[np.random.randint(len(istacks))])
            if len(bad_stacks) >= self.size:
                break
        stacks = np.vstack([good_stacks, np.array(bad_stacks)])
        stacks = np.random.permutation(stacks)[:self.size]
        return tuple(map(list, zip(*stacks)))
