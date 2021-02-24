from typing import List, Tuple, Dict, Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from tqdm import tqdm

from evaluation.stack_sim import metric_in_time, bootstrap_bin_metric


def bootstrap_aggregate_metric(metric: Callable[[List[float]], float], y: List[float],
                               err: float = 0.05, iters: int = 100, size: int = 1):
    values = []
    y = np.array(y)
    real_value = metric(y)
    n = len(y)
    sn = int(size * n)
    left = int(iters * err / 2)
    while len(values) < iters:
        inds = np.random.choice(n, sn)
        try:
            value = metric(y[inds])
            values.append(value)
        except:
            pass
    values = sorted(values)
    # return round(real_value, 4), round(values[left], 4), round(values[iters - 1 - left], 4)
    return real_value, values[left], values[iters - 1 - left]


def plot_prec_rec_curve(y_true, y_pred):
    precision, recall, thrs = precision_recall_curve(y_true, y_pred)

    ax = plt.subplot(221)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")


def draw_prec_rec_at_time(y_true, y_pred, th=0):
    prec, l_prec, r_prec = metric_in_time(y_true, y_pred >= th, precision_score, with_ci=True)
    rec, l_rec, r_rec = metric_in_time(y_true, y_pred >= th, recall_score, with_ci=True)
    x = range(len(prec))

    ax = plt.subplot(222)

    ax.plot(x, prec, color='b', label='precision')
    ax.fill_between(x, l_prec, r_prec, color='b', alpha=.1)

    ax.plot(x, rec, color='r', label='recall')
    ax.fill_between(x, l_rec, r_rec, color='r', alpha=.1)

    ax.set_ylim((0, 1.1))
    plt.legend(loc="lower right")

    return prec, rec


def transform_pred(preds, k=1):
    y_true, y_pred = [], []
    for y_tr, prs in preds:
        sorted_pr = sorted(prs.items(), key=lambda x: -x[1])
        i = 0
        top_pr = []
        while i < len(sorted_pr) and (i <= k or sorted_pr[i] == sorted_pr[0]):
            top_pr.append(sorted_pr[i])
            i += 1
        y_true.append(y_tr in set(p[0] for p in top_pr))
        y_pred.append(top_pr[-1][1])

    return np.array(y_true), np.array(y_pred)


def score_model(preds: List[Tuple[int, Dict[int, float]]], th=None, k=1, full=True, model_name=None):
    th = th or float("-inf")
    y_true, y_pred = transform_pred(preds, k)

    prec, l_prec, r_prec = bootstrap_bin_metric(y_true, y_pred >= th, precision_score)
    rec, l_rec, r_rec = bootstrap_bin_metric(y_true, y_pred >= th, recall_score)

    if not full:
        return prec, rec

    print(f"Precision: {prec} ({l_prec}, {r_prec}), Recall: {rec} ({l_rec}, {r_rec})")
    plt.figure(figsize=(15, 10))
    if model_name is not None:
        plt.suptitle(model_name)
    plot_prec_rec_curve(y_true, y_pred)
    time_prec, time_rec = draw_prec_rec_at_time(y_true, y_pred, th)
    plt.show()
    return prec, rec, (time_prec, time_rec)


def auc_metric(preds: List[Tuple[int, Dict[int, float]]]) -> float:
    aps = []
    for true_is_id, is_scores in preds:
        sorted_scores = sorted(is_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (is_id, score) in enumerate(sorted_scores):
            if is_id == true_is_id:
                aps.append(1 / (i + 1))
                break
    return float(np.mean(aps))


def map_metric(preds: Iterable[Tuple[int, Dict[int, float]]]) -> float:
    aps = []
    for true_is_id, is_scores in preds:
        sorted_scores = sorted(is_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (is_id, score) in enumerate(sorted_scores):
            if is_id == true_is_id:
                aps.append(1 / (i + 1))
                break
    return float(np.mean(aps))


def recall_rate(preds: Iterable[Tuple[int, Dict[int, float]]], k: int = 1) -> float:
    correct_top = []
    for true_is_id, is_scores in preds:
        sorted_scores = sorted(is_scores.items(), key=lambda x: x[1], reverse=True)
        found_at = -1
        for i, (is_id, score) in enumerate(sorted_scores):
            if is_id == true_is_id:
                found_at = i
                break
        if found_at >= 0:
            correct_top.append(found_at < k)
    return float(np.mean(correct_top))


def paper_metrics(preds: List[Tuple[int, Dict[int, float]]]) -> Dict[str, float]:
    scores = {}
    scores["map"] = map_metric(preds)
    scores["rr@1"] = recall_rate(preds, 1)
    scores["rr@5"] = recall_rate(preds, 5)
    scores["rr@10"] = recall_rate(preds, 10)

    for name, score in scores.items():
        print(f"{name}: {score}")
    return scores


def paper_metrics_iter(preds: Iterable[Tuple[int, Dict[int, float]]]) -> Dict[str, float]:
    aps = []
    correct_top = []
    total_preds = 0
    for true_is_id, is_scores in tqdm(preds):
        total_preds += 1
        sorted_scores = sorted(is_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (is_id, score) in enumerate(sorted_scores):
            if is_id == true_is_id:
                aps.append(1 / (i + 1))
                correct_top.append(i)
                break

    scores = {}
    scores["mrr"] = bootstrap_aggregate_metric(np.mean, aps)
    scores["rr@1"] = bootstrap_aggregate_metric(np.mean, [x < 1 for x in correct_top])
    scores["rr@5"] = bootstrap_aggregate_metric(np.mean, [x < 5 for x in correct_top])
    scores["rr@10"] = bootstrap_aggregate_metric(np.mean, [x < 10 for x in correct_top])

    for name, score in scores.items():
        print(f"{name}: {score}")
    return scores
