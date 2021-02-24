from abc import ABC, abstractmethod
from typing import Optional, Any, List

import torch
from torch import nn
import torch.nn.functional as F

from data.buckets.issues_data import StackAdditionState
from data.pair_sim_selector import PairSimSelector
from data.triplet_selector import TripletSelector
from methods.neural import device
from methods.neural.neural_base import NeuralModel


class LossComputer(ABC):
    @abstractmethod
    def get_event(self, event: StackAdditionState) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def get_raw(self, *args, **kwargs) -> torch.Tensor:
        pass

    def get_eval_raws(self, rows: List[Any]) -> float:
        with torch.no_grad():
            return sum(self.get_raw(*row) for row in rows).data.cpu().item() / len(rows)


class PointLossComputer(LossComputer):
    def __init__(self, model: NeuralModel, pair_sim_selector: PairSimSelector = None):
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()
        self.pair_sim_selector = pair_sim_selector

    def get_event(self, event: StackAdditionState) -> Optional[torch.Tensor]:
        stack_ids, target = self.pair_sim_selector(event)
        if len(stack_ids) == 0:
            return None
        sim = torch.cat([self.model(event.st_id, sid).view(1, -1) for sid in stack_ids])  # .view(1, -1)
        target = torch.tensor(target).to(device)
        return self.loss_function(sim, target)

    def get_raw(self, id1: int, id2: int, label: int) -> torch.Tensor:
        sim = self.model(id1, id2).view(1, -1)
        target = torch.tensor([label]).to(device)
        return self.loss_function(sim, target)


class PairLossComputer(LossComputer, ABC):
    def __init__(self, model: NeuralModel, triplet_selector: TripletSelector, loss_function):
        self.model = model
        self.loss_function = loss_function
        self.triplet_selector = triplet_selector

    def get_event_predictions(self, event: StackAdditionState):
        good_stacks, bad_stacks = self.triplet_selector(event)
        if len(good_stacks) == 0:
            return None
        predictions = []
        for good_id, bad_id in zip(good_stacks, bad_stacks):
            good_sim = self.model(event.st_id, good_id)
            bad_sim = self.model(event.st_id, bad_id)
            predictions.append((good_sim - bad_sim).view(1, -1))
        return torch.cat(predictions)

    def get_raw_predictions(self, st_id: int, good_id: int, bad_id: int) -> torch.Tensor:
        good_sim = self.model(st_id, good_id)
        bad_sim = self.model(st_id, bad_id)
        return (good_sim - bad_sim).view(1, -1)


class RanknetLossComputer(PairLossComputer):
    def __init__(self, model: NeuralModel, triplet_selector: TripletSelector = None):
        super().__init__(model, triplet_selector, nn.CrossEntropyLoss())

    def get_event(self, event: StackAdditionState) -> Optional[torch.Tensor]:
        predictions = self.get_event_predictions(event)
        if predictions is None:
            return None

        target = torch.tensor([1] * len(predictions)).to(device)
        return self.loss_function(predictions, target)

    def get_raw(self, st_id: int, good_id: int, bad_id: int) -> torch.Tensor:
        predictions = self.get_raw_predictions(st_id, good_id, bad_id)
        target = torch.tensor([1] * len(predictions)).to(device)
        return self.loss_function(predictions, target)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, diff):
        losses = F.relu(diff + self.margin)
        return losses.mean()


class TripletLossComputer(PairLossComputer):
    def __init__(self, model: NeuralModel, triplet_selector: TripletSelector, margin: float = 0.2):
        super().__init__(model, triplet_selector, TripletLoss(margin))

    def get_event(self, event: StackAdditionState) -> Optional[torch.Tensor]:
        predictions = self.get_event_predictions(event)
        if predictions is None:
            return None
        return self.loss_function(predictions[:, 0])

    def get_raw(self, st_id: int, good_id: int, bad_id: int) -> torch.Tensor:
        predictions = self.get_raw_predictions(st_id, good_id, bad_id)
        return self.loss_function(predictions[:, 0])
