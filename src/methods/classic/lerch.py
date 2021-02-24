from typing import List, Tuple

from methods.base import SimStackModel
from methods.classic.tfidf import IntTfIdfComputer
from preprocess.seq_coder import SeqCoder


class LerchModel(SimStackModel):
    def __init__(self, coder: SeqCoder):
        self.tfidf = IntTfIdfComputer(coder)

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'LerchModel':
        self.tfidf.fit(unsup_data)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        anchor_tfidf = self.tfidf.transform(anchor_id)

        max_score = 1
        scores = []
        for stack_id in stack_ids:
            score = 0
            for word in self.tfidf.words_tfs(stack_id).keys():
                if word not in anchor_tfidf:
                    continue
                tf, idf = anchor_tfidf[word]
                tf_idf_pow2 = tf * idf ** 2
                score += tf_idf_pow2

            denom = 1
            scores.append(score / max_score / denom)

        return scores

    def name(self) -> str:
        return self.tfidf.name() + "_lerch"
