import json
import os
from typing import List, Tuple

import pandas as pd

from data.objects import Stack


def read_stack(path: str, frames_field: str = 'frames') -> Stack:
    with open(path) as f:
        dict = json.loads(f.read())
        return Stack(dict['id'], dict['timestamp'], dict['class'], dict.get(frames_field, dict["frames"])[0],
                     dict.get('message', None), dict.get('comment', None))


def dir_stacks_id(dir_path: str, size: int = -1) -> List[int]:
    file_names = os.listdir(dir_path)
    if size > 0:
        file_names = file_names[:size]
    return [int(name[:-5]) for name in file_names]


def read_supervised(path: str, have_train_indicator: bool = False, verbose: bool = False) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    df = pd.read_csv(path)
    target_tr, target_te = [], []
    for row in df.itertuples():
        t = (row.rid1, row.rid2, int(row.label))
        if have_train_indicator and row.train:
            target_tr.append(t)
        else:
            target_te.append(t)
    if verbose:
        print(f"Train pairs count: {len(target_tr)}")
        print(f"Test pairs count: {len(target_te)}")
    return target_tr, target_te


def read_pairs(path: str) -> List[Tuple[int, int, int]]:
    df = pd.read_csv(path)
    target = []
    for row in df.itertuples():
        t = (row.rid1, row.rid2, int(row.label))
        target.append(t)
    return target


def sim_data_stack_ids(sim_data: List[Tuple[int, int, int]]) -> List[int]:
    return list(set(p[0] for p in sim_data) | set(p[1] for p in sim_data))


def to_unsup(sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> List[int]:
    train_stacks = []
    if sim_train_data is not None:
        train_stacks += sim_data_stack_ids(sim_train_data)
    if unsup_data is not None:
        train_stacks += unsup_data
    return train_stacks
