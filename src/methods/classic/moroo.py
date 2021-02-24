from typing import List, Tuple, Dict

from methods.classic.hyperopt import SimStackHyperoptModel
from methods.classic.lerch import LerchModel
from methods.classic.rebucket import RebucketModel
from preprocess.seq_coder import SeqCoder


class MorooModel(SimStackHyperoptModel):
    def __init__(self, coder: SeqCoder, alpha: float = 1.0, c: float = 0.0, o: float = 0.0):
        self.coder = coder
        self.rebucket = RebucketModel(coder, c, o)
        self.lerch = LerchModel(coder)
        self.alpha = alpha

    def set_params(self, args: Dict[str, float]):
        self.rebucket.c = args['c']
        self.rebucket.o = args['o']
        self.alpha = args['alpha']

    def params_edges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "c": (0, 2),
            "o": (0, 2),
            'alpha': (0, 1)
        }

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'MorooModel':
        self.rebucket.fit(sim_train_data, unsup_data)
        self.lerch.fit(sim_train_data, unsup_data)

        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        lens = [len(self.lerch.tfidf.words_tfs(stack_id)) for stack_id in stack_ids]
        rebucket_score = self.rebucket.predict(anchor_id, stack_ids)
        lerch_score = self.lerch.predict(anchor_id, stack_ids)
        lerch_score = [score / (l ** 0.5) for score, l in zip(lerch_score, lens)]
        return [r * l / (self.alpha * r + (1 - self.alpha) * l) for r, l in zip(rebucket_score, lerch_score)]

    def name(self) -> str:
        return self.coder.name() + f"_moroo_{self.rebucket.c}_{self.rebucket.o}_{self.alpha}"

    def save(self, name: str = ""):
        self.lerch.save(name)

    def load(self, name: str = "") -> 'MorooModel':
        lerch = self.lerch.load(name)
        if lerch is not None:
            self.lerch = lerch
        return self
