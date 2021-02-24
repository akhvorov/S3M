import json
import os

from functools import lru_cache

from abc import ABC, abstractmethod

from data.objects import Stack
from data.readers import read_stack


class StackLoader(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, id: int) -> Stack:
        raise NotImplementedError


class DirectoryStackLoader(StackLoader):
    def __init__(self, *dirs: str, frames_field: str = 'frames'):
        self.dirs = list(dirs)
        self.id_dir = {}
        self.frames_field = frames_field

    def init(self, *dirs: str):
        self.dirs += list(dirs)

    def add(self, directory: str):
        self.dirs.append(directory)

    def name(self) -> str:
        return ("rec" if self.frames_field == "frames" else "notrec") + "_loader"

    @lru_cache(maxsize=300_000)
    def __call__(self, id: int) -> Stack:
        if id not in self.id_dir:
            for d in self.dirs:
                if os.path.exists(f"{d}/{id}.json"):
                    self.id_dir[id] = d
                    break
        if id in self.id_dir:
            return read_stack(f"{self.id_dir[id]}/{id}.json", self.frames_field)
        return None


class JsonStackLoader(StackLoader):
    def __init__(self, reports_path: str):
        self.reports_path = reports_path
        self.reports = {}

        raw_reports = json.load(open(reports_path, 'r'))
        for report in raw_reports:
            if report is None:
                continue

            stacktrace = report["stacktrace"]
            if isinstance(stacktrace, list):
                stacktrace = stacktrace[0]
            st_id = report["bug_id"]
            exception = stacktrace["exception"] or []
            if isinstance(exception, str):
                exception = [exception]

            raw_frames = stacktrace["frames"]
            frames = [frame["function"] for frame in raw_frames]

            self.reports[st_id] = Stack(st_id, report["creation_ts"], exception, frames)

    def name(self) -> str:
        return "json_loader"

    def __call__(self, id: int) -> Stack:
        return self.reports[id]
