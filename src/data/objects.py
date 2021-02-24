from collections import namedtuple
from typing import List


class Stack:
    def __init__(self, id: int, timestamp: int, clazz: List[str], frames: List[str],
                 message: List[str] = None, comment: str = None):
        self.id = id
        self.ts = timestamp
        self.clazz = clazz
        self.frames = frames
        self.message = message or []
        self.comment = comment or ""

    def eq_content(self, stack: 'Stack'):
        return self.clazz == stack.clazz and self.frames == stack.frames and \
               self.message == stack.message and self.comment == stack.comment

    @property
    def is_soe(self) -> bool:
        return self.clazz and max('StackOverflow' in cl for cl in self.clazz)


StackEvent = namedtuple('StackEvent', 'id ts label')


class Issue:
    def __init__(self, id: int, ts: int):
        self.id = id
        self.stacks = {}
        self.last_update_ts = [ts]

    def add(self, st_id: int, ts: int, label: bool):
        if st_id in self.stacks:
            raise ValueError("stack already in this issue")
        self.stacks[st_id] = StackEvent(st_id, ts, label)
        self.last_update_ts.append(ts)

    def remove(self, st_id: int, ts: int, label: bool):
        self.last_update_ts.remove(self.stacks[st_id].ts)
        del self.stacks[st_id]

    def confident_state(self) -> List[StackEvent]:
        #         return [st[0] for st in self.stacks.values() if st[2]]
        return list(self.stacks.values())

    def last_ts(self) -> int:
        return self.last_update_ts[-1]
