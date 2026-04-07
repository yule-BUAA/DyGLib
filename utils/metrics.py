import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_mrr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for link prediction.
    :param pos_scores: numpy array of positive edge scores, shape (num_pos_samples,)
    :param neg_scores: numpy array of negative edge scores, shape (num_neg_samples,)
    :return: MRR score
    """
    y_pred_pos = pos_scores.reshape(-1, 1)
    y_pred_neg = neg_scores.reshape(-1, 1)

    # Compute ranking: how many negatives score higher than each positive
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1.0 / ranking_list.astype(np.float32)
    return float(mrr_list.mean())


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, ) - concatenated positive and negative predictions
    :param labels: Tensor, shape (num_samples, ) - concatenated labels (1s for positive, 0s for negative)
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    # Split predictions into positive and negative for MRR calculation
    # DyGLib concatenates [positive_probs, negative_probs]
    num_pos = int(labels.sum())
    num_neg = len(labels) - num_pos

    if num_pos > 0 and num_neg > 0:
        pos_scores = predicts[:num_pos]
        neg_scores = predicts[num_pos:]
        mrr = compute_mrr(pos_scores, neg_scores)
    else:
        mrr = 0.0

    return {'average_precision': average_precision, 'roc_auc': roc_auc, 'mrr': mrr}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
