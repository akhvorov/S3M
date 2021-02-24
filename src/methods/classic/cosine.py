from typing import List, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from methods.base import SimStackModel
from methods.classic.tfidf import IntTfIdfComputer
from preprocess.seq_coder import SeqCoder


class CosineModel(SimStackModel):
    def __init__(self, coder: SeqCoder):
        self.tfidf = IntTfIdfComputer(coder)
        self.coder = coder

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'CosineModel':
        self.tfidf.fit(unsup_data)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []

        def to_dict(stack_id):
            # stack_freqs = Counter(self.coder(stack_id))
            # return stack_freqs
            # return {frame: 1 for frame, cnt in stack_freqs.items()}
            # return {word: v[0] * v[1] ** 2 for word, v in self.tfidf_model.tfidf_vectorizer.transform(stack_id).items()}
            return {word: v[1] for word, v in self.tfidf.transform(stack_id).items()}

        anchor = to_dict(anchor_id)
        for stack_id in stack_ids:
            array = DictVectorizer().fit_transform([anchor, to_dict(stack_id)])
            scores.append(cosine_similarity(array[0:], array[1:])[0, 0])

        return scores

    def name(self) -> str:
        return self.coder.name() + "_cosine"
