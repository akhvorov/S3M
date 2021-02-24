from typing import Iterable, List

import os

from data.buckets.bucket_data import BucketData, StackAdditionEvent
from data.buckets.event_state_model import EventStateModel, StackAdditionState


class BucketDataset:
    def __init__(self, bucket_data: BucketData):
        self.name = bucket_data.name
        self.events = bucket_data.events()

        self.warmup_days = bucket_data.warmup_days
        self.train_days = bucket_data.train_days
        self.val_days = bucket_data.val_days
        self.test_days = bucket_data.test_days

        self._train_stacks = None

        self.train_done = False
        self.test_done = False

    def time_slice_events(self, start: float, finish: float) -> List[StackAdditionEvent]:
        return [event for event in self.events if start <= event.ts < finish]

    def _cached_event_state(self, until_day: float = None) -> EventStateModel:
        if until_day is None:
            return EventStateModel(self.name, self.warmup_days)
        event_model = EventStateModel(self.name, self.warmup_days)  # , self.warmup_days remove it for final state for irving

        if os.path.exists(event_model.file_name(until_day)):
            event_model.load(until_day)
        else:
            load_prev = False
            for i in range(int(until_day), 0, -1):
                if os.path.exists(event_model.file_name(i)):
                    event_model.load(i)
                    event_model.warmup(self.time_slice_events(i, until_day))
                    load_prev = True
                    print(f"post train from {i} to {until_day}")
                    break
            if not load_prev:
                event_model.warmup(self.time_slice_events(0, until_day))

            event_model.save(until_day)

        return event_model

    def reset(self) -> 'BucketDataset':
        self.train_done = False
        self.test_done = False
        return self

    def train_stacks(self) -> List[int]:
        if self._train_stacks is None:
            event_model = self._cached_event_state(self.warmup_days + self.train_days + self.val_days)
            self._train_stacks = list(event_model.all_seen_stacks())
            self._train_stacks = sorted(self._train_stacks)
        return self._train_stacks

    def generate_events(self, start: float, longitude: float) -> Iterable[StackAdditionState]:
        event_model = self._cached_event_state(start)
        return event_model.collect(self.time_slice_events(start, start + longitude))

    def train(self) -> Iterable[StackAdditionState]:
        return self.generate_events(self.warmup_days, self.train_days)

    def validation(self) -> Iterable[StackAdditionState]:
        return self.generate_events(self.warmup_days + self.train_days, self.val_days)

    def test(self) -> Iterable[StackAdditionState]:
        return self.generate_events(self.warmup_days + self.train_days + self.val_days, self.test_days)
