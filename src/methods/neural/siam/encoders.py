from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from methods.neural import device
from preprocess.seq_coder import SeqCoder


class Encoder(ABC, nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def opt_params(self) -> List[torch.tensor]:
        return []

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class EncoderModel(Encoder):
    def __init__(self, name: str, coder: SeqCoder, dim: int, out_dim: int = None, **kvargs):
        super(EncoderModel, self).__init__()
        self.coder = coder
        self.word_embeddings = nn.Embedding(len(coder), dim)
        self.dim = dim
        self._out_dim = out_dim or dim
        self._name = coder.name() + "_" + name + f"_rand_dim={dim}"

    def to_inds(self, stack_id: int, reverse: bool = False) -> torch.tensor:
        if reverse:
            return torch.tensor(self.coder(stack_id)[::-1]).to(device)
        return torch.tensor(self.coder(stack_id)).to(device)

    def out_dim(self) -> int:
        return self._out_dim

    def opt_params(self) -> List[torch.tensor]:
        return []

    def name(self) -> str:
        return self._name

    def train(self, mode: bool = True):
        super().train(mode)
        self.coder.train(mode)


class LSTMEncoder(EncoderModel):
    def __init__(self, coder: SeqCoder, dim: int = 50, hid_dim: int = 200, bidir: bool = True, **kvargs):
        super(LSTMEncoder, self).__init__(f"lstm_frames.bidir={bidir},hdim={hid_dim}",
                                          coder, dim, out_dim=hid_dim, **kvargs)
        self.hidden_dim = hid_dim // 2 if bidir else hid_dim
        self.bidir = bidir
        self.lstm_forward = nn.LSTM(dim, self.hidden_dim)
        self.lstm_backward = nn.LSTM(dim, self.hidden_dim)

    def forward(self, stack_id: int) -> torch.tensor:
        emb_f = self.word_embeddings(self.to_inds(stack_id))
        lstm_f_out, hidden_f = self.lstm_forward(emb_f.view(emb_f.shape[0], 1, -1))
        out = lstm_f_out[-1][0]
        if self.bidir:
            emb_b = self.word_embeddings(self.to_inds(stack_id, reverse=True))
            lstm_b_out, hidden_b = self.lstm_backward(emb_b.view(emb_b.shape[0], 1, -1))
            out = torch.cat((out, lstm_b_out[-1][0]))
        return out

    def opt_params(self) -> List[torch.tensor]:
        return list(self.lstm_forward.parameters()) + (list(self.lstm_backward.parameters()) if self.bidir else [])
