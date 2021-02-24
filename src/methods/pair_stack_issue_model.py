from typing import Dict, Tuple, Iterable, Union

from data.buckets.issues_data import StackAdditionState
from data.objects import Issue
from methods.base import SimIssueModel, SimStackModel, IssueScorer


class MaxIssueScorer(IssueScorer):
    def score(self, scores: Iterable[float], with_arg: bool = False) -> Union[float, Tuple[float, int]]:
        if with_arg:
            ind = 0
            value = None
            for i, score in enumerate(scores):
                if value is None or score > value:
                    ind = i
                    value = score
            return value, ind
        return max(scores)

    def name(self) -> str:
        return "max_scorer"


class PairStackBasedSimModel(SimIssueModel):
    def __init__(self, stack_model: SimStackModel, issue_scorer: IssueScorer):
        self.stack_model = stack_model
        self.issue_scorer = issue_scorer

    def name(self) -> str:
        return "_".join([model.name() for model in [self.stack_model, self.issue_scorer]])

    def predict_all(self, st_id: int, issues: Dict[int, Issue], with_stacks: bool = False) \
            -> Tuple[Dict[int, Union[float, Tuple[float, int]]], int]:
        pred_issues = {}
        stacks_cnt = 0
        for id, issue in issues.items():
            # stacks = self.stacks_selector.select(issue.confident_state())
            stacks = issue.confident_state()
            stacks_cnt += len(stacks)
            if len(stacks) == 0:
                pred_issues[id] = 0
                continue
            preds = self.stack_model.predict(st_id, [st.id for st in stacks])
            score = self.issue_scorer.score(preds, with_arg=False)
            pred_issues[id] = score
        return pred_issues, stacks_cnt

    def predict(self, events: Iterable[StackAdditionState]) -> Iterable[Tuple[int, Dict[int, float]]]:
        for i, event in enumerate(events):
            pred_issues, _ = self.predict_all(event.st_id, event.issues)
            yield event.is_id, pred_issues
