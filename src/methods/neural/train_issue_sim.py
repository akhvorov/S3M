import copy
from itertools import islice
from time import time
from typing import List, Tuple

import torch

from data.buckets.issues_data import BucketDataset, StackAdditionState
from data.pair_sim_selector import RandomPairSimSelector
from data.triplet_selector import RandomTripletSelector
from evaluation.issue_sim import score_model, paper_metrics_iter
from methods.neural.losses import PointLossComputer, LossComputer, TripletLossComputer, RanknetLossComputer
from methods.neural.neural_base import NeuralModel
from methods.pair_stack_issue_model import PairStackBasedSimModel, MaxIssueScorer


def log_metrics(sim_stack_model: NeuralModel, loss_computer: LossComputer,

                train_sim_pairs_data_for_score: List[Tuple[int, int, int]],
                test_sim_pairs_data_for_score: List[Tuple[int, int, int]],

                train_data_for_score: List[StackAdditionState],
                test_data_for_score: List[StackAdditionState],

                prefix: str, writer, n_iter: int):
    with torch.no_grad():
        train_loss_value = loss_computer.get_eval_raws(train_sim_pairs_data_for_score)
        test_loss_value = loss_computer.get_eval_raws(test_sim_pairs_data_for_score)

        ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer())
        train_preds = ps_model.predict(train_data_for_score)
        test_preds = ps_model.predict(test_data_for_score)
        train_score = score_model(train_preds, full=False)
        test_score = score_model(test_preds, full=False)
    print(prefix +
          f"Train loss: {round(train_loss_value, 4)}. "
          f"Test loss: {round(test_loss_value, 4)}. "
          f"Train prec {train_score[0]}, rec {train_score[1]}. "
          f"Test prec {test_score[0]}, rec {test_score[1]}       ", end='')

    if writer:
        writer.add_scalar('Loss/train', train_loss_value, n_iter)
        writer.add_scalar('Loss/test', test_loss_value, n_iter)


def log_all_data_scores(sim_stack_model: NeuralModel, data_gen):
    data_gen.reset()
    ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer())  # =None for no filter

    # train_preds = ps_model.predict(data_gen.train())
    # print("Train")
    # # score_model(train_preds, th=-1, model_name="Train " + sim_stack_model.name())
    # paper_metrics_iter(train_preds)

    # test_preds = ps_model.predict(data_gen.test())
    test_preds = ps_model.predict(data_gen.test())
    print("Test")
    # te_score = score_model(test_preds, model_name="Test " + sim_stack_model.name())
    paper_metrics_iter(test_preds)

    print()


def train_issue_model(sim_stack_model: NeuralModel, data_gen: BucketDataset,
                      loss_name: str, optimizers: List,
                      epochs: int = 1, batch_size: int = 25, selection_from_event_num: int = 4,
                      writer=None, period: int = 25):
    if loss_name == "point":
        train_selector = RandomPairSimSelector(selection_from_event_num)
        loss_computer = PointLossComputer(sim_stack_model, train_selector)
    elif loss_name == "ranknet":
        train_selector = RandomTripletSelector(selection_from_event_num)
        loss_computer = RanknetLossComputer(sim_stack_model, train_selector)
    elif loss_name == "triplet":
        train_selector = RandomTripletSelector(selection_from_event_num)
        loss_computer = TripletLossComputer(sim_stack_model, train_selector, margin=0.2)
    else:
        raise ValueError

    train_data_for_score = [copy.deepcopy(x) for x in islice(data_gen.train(), 50)]
    test_data_for_score = [copy.deepcopy(x) for x in islice(data_gen.test(), 50)]
    train_sim_pairs_data_for_score = list(train_selector.generate(train_data_for_score))
    test_sim_pairs_data_for_score = list(train_selector.generate(test_data_for_score))
    data_gen.reset()
    assert len(train_sim_pairs_data_for_score) > 0

    start = time()
    print("Time to score validation data:", time() - start)

    n_iter = 0
    for epoch in range(epochs):
        data_gen.reset()
        for i, event in enumerate(data_gen.train()):
            sim_stack_model.train(True)
            loss = loss_computer.get_event(event)
            if loss is None:
                continue
            loss.backward()

            if i != 0 and i % batch_size == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

            sim_stack_model.train(False)
            if i == 0 or (i + 1) % period == 0:
                prefix = f"\rEpoch {epoch}: {i + 1}. "
                log_metrics(sim_stack_model, loss_computer,
                            train_sim_pairs_data_for_score, test_sim_pairs_data_for_score,
                            train_data_for_score, test_data_for_score,
                            prefix, writer, n_iter)
            if (i + 1) % 1000 == 0:
                print()
            n_iter += 1
        print()
        print(f"Epoch {epoch} done.")

        log_all_data_scores(sim_stack_model, data_gen)
    if writer:
        writer.close()
    return sim_stack_model
