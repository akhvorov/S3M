from abc import ABC
from typing import List, Tuple, Iterable

from torch import nn

from methods.base import SimStackModel


class NeuralModel(nn.Module, SimStackModel, ABC):
    def __init__(self):
        super(NeuralModel, self).__init__()

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: Iterable[int] = None) -> 'NeuralModel':
        return self

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return self.agg.opt_params() + self.classifier.opt_params()
