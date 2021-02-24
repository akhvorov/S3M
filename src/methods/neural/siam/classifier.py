from typing import List

import torch
from torch import nn


class StackClassifier(nn.Module):
    def __init__(self, input_dim: int = 50, num_layers: int = 2, out_num: int = 1,
                 features_num: int = 5, dropout: float = 0.1):
        super(StackClassifier, self).__init__()
        self.features_num = features_num
        self.input_dim = features_num * input_dim
        self.num_layers = num_layers
        self.out_num = out_num
        self.dropout = dropout
        if num_layers == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.input_dim, out_num)
            )
        elif num_layers == 2:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.input_dim, int(self.input_dim / 2)),
                nn.ReLU(),
                nn.Linear(int(self.input_dim / 2), out_num),
            )

    def forward(self, v1: torch.tensor, v2: torch.tensor) -> torch.tensor:
        if self.num_layers == 0:
            cl = torch.exp(-(v1 - v2).norm(p=2, dim=0)).view(1, 1)
            return torch.cat((1 - cl, cl))

        diff = torch.abs(v1 - v2)
        if self.features_num == 1:
            features = diff
        elif self.features_num == 2:
            features = torch.cat((diff, (v1 + v2) / 2), 0)
        elif self.features_num == 3:
            features = torch.cat((diff, (v1 + v2) / 2, v1 * v2), 0)
        elif self.features_num == 4:
            features = torch.cat((diff, v1, v2, v1 * v2), 0)
        elif self.features_num == 5:
            features = torch.cat((diff, v1, v2, (v1 + v2) / 2, v1 * v2), 0)
        else:
            raise ValueError("Wrong features_num parameter value")

        cl = self.classifier(features)
        if self.out_num == 1:
            cl = torch.cat((1 - cl, cl))
        else:
            cl = nn.functional.softmax(cl, dim=0)
        return cl

    def opt_params(self) -> List[torch.tensor]:
        return list(self.classifier.parameters())

    def name(self) -> str:
        return "dssm." + ".".join([f"cl_layers={self.num_layers}",
                                   f"out_num={self.out_num}",
                                   f"fnum={self.features_num}",
                                   f"cl_do={self.dropout}"])
