from typing import List

import torch
from torch import nn

from methods.neural.siam.encoders import EncoderModel, Encoder


class EncodersAggregation(Encoder):
    def __init__(self, name: str, encoders: List[EncoderModel]):
        super(EncodersAggregation, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self._name = name

    def encode(self, stack_id: int) -> List[torch.tensor]:
        return [encoder(stack_id) for encoder in self.encoders]

    def out_dim(self) -> int:
        raise NotImplementedError

    def opt_params(self) -> List[torch.tensor]:
        res = []
        for encoder in self.encoders:
            res += encoder.opt_params()
        return res

    def name(self) -> str:
        return "__".join(encoder.name() for encoder in self.encoders) + "__" + self._name

    def train(self, mode: bool = True):
        super().train(mode)


class ConcatAggregation(EncodersAggregation):
    def __init__(self, encoders: List[EncoderModel]):
        super(ConcatAggregation, self).__init__("concat_agg", encoders)
        self._out_dim = sum(encoder.out_dim() for encoder in encoders)

    def forward(self, stack_id: int) -> torch.tensor:
        return torch.cat(self.encode(stack_id), axis=0)

    def out_dim(self) -> int:
        return self._out_dim

    def opt_params(self) -> List[torch.tensor]:
        return super().opt_params()
