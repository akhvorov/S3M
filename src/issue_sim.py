from time import time
from typing import Optional

import torch

from data.buckets.bucket_data import BucketData
from data.buckets.issues_data import BucketDataset
from evaluation.issue_sim import paper_metrics_iter
from methods.classic.hyperopt import PairStackBasedIssueHyperoptModel
from methods.neural.train_issue_sim import train_issue_model
from methods.pair_stack_issue_model import MaxIssueScorer, PairStackBasedSimModel
from models_factory import create_neural_model, create_classic_model
from utils import set_seed, random_seed


def classic_issues(bucket_data: BucketData, method: str,
                   max_len: Optional[int] = None, trim_len: int = 0):
    set_seed(random_seed)
    print("Dataset:", bucket_data.name)
    print("train_days", bucket_data.train_days, "test_days", bucket_data.test_days,
          "warmup_days", bucket_data.warmup_days, "val_days", bucket_data.val_days)
    bucket_data.load()
    print("Load data")
    stack_loader = bucket_data.stack_loader()
    dataset = BucketDataset(bucket_data)
    unsup_stacks = dataset.train_stacks()
    print("Get train stacks")

    start = time()

    model = create_classic_model(stack_loader, method, max_len, trim_len, bucket_data.sep)
    print("Create model")
    model.fit([], unsup_stacks)
    print("Fit model")
    ps_model = PairStackBasedIssueHyperoptModel(model, MaxIssueScorer())
    ps_model.find_params(dataset)
    print("Find params")
    print("Time to fit", time() - start)

    start = time()
    new_preds = ps_model.predict(dataset.test())
    # ps_model.dump_issue_scores(dataset.test(), "../data/scores_in_issue_hist_70-70-400.json")
    print("Time to predict", time() - start)

    start = time()
    # score_model(new_preds, full=True, model_name=model.name())
    # draw_acc_at_th(new_preds, model_name=model.name())
    paper_metrics_iter(new_preds)
    print("Time to eval", time() - start)


def neural_issues(bucket_data: BucketData,
                  max_len: Optional[int] = None, trim_len: int = 0,
                  loss_name: str = 'point', hyp_top_stacks: int = 20, hyp_top_issues: int = 5,
                  epochs: int = 1):
    set_seed(random_seed)
    print("Dataset:", bucket_data.name)
    print("train_days", bucket_data.train_days, "test_days", bucket_data.test_days,
          "warmup_days", bucket_data.warmup_days, "val_days", bucket_data.val_days, 'loss_name', loss_name)
    bucket_data.load()
    stack_loader = bucket_data.stack_loader()
    dataset = BucketDataset(bucket_data)
    unsup_stacks = dataset.train_stacks()

    model = create_neural_model(stack_loader, unsup_stacks, max_len, trim_len)

    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)]
    train_issue_model(model, dataset, loss_name, optimizers,
                      epochs=epochs, batch_size=10, period=100, selection_from_event_num=4, writer=None)

    ps_model = PairStackBasedSimModel(model, MaxIssueScorer())

    start = time()
    new_preds = ps_model.predict(dataset.test())
    print("Time to predict", time() - start)

    # draw_acc_at_th(new_preds, model_name=model.name())
    # paper_metrics_iter(new_preds)
