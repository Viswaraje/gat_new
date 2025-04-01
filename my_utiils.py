import math
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Tensor Initialization Functions
def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)

def reset(nn):
    def _reset(item):
        if hasattr(item, "reset_parameters"):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, "children") and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

# Evaluation Metrics
def metrics_graph(yt, yp):
    """
    Calculate performance metrics for graph-based predictions.

    Args:
        yt: Ground truth labels (numpy array)
        yp: Predicted probabilities (numpy array)

    Returns:
        auc: Area Under ROC Curve
        aupr: Area Under Precision-Recall Curve
        f1_score: F1 Score at optimal threshold
        accuracy: Accuracy at optimal threshold
    """
    yt = np.array(yt).flatten()
    yp = np.array(yp).flatten()

    # AUC & AUPR Calculation
    aupr = average_precision_score(yt, yp)
    auc = roc_auc_score(yt, yp)

    # Define threshold range using np.linspace
    thresholds = np.linspace(yp.min(), yp.max(), 1000)

    # Compute confusion matrix elements at each threshold
    predict_score_matrix = np.tile(yp, (len(thresholds), 1)) >= thresholds[:, None]
    TP = (predict_score_matrix @ yt[:, None]).flatten()
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = yt.sum() - TP
    TN = len(yt) - TP - FP - FN

    # Calculate F1-score and Accuracy
    f1_score_list = np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP, dtype=float), where=(2 * TP + FP + FN) != 0)
    accuracy_list = np.divide(TP + TN, TP + TN + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + TN + FP + FN) != 0)

    # Find optimal threshold based on F1-score
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]

    # Log confusion matrix details at optimal threshold
    logging.info(f"Confusion Matrix at Optimal Threshold:")
    logging.info(f"TP: {TP[max_index]}, FP: {FP[max_index]}, FN: {FN[max_index]}, TN: {TN[max_index]}")
    logging.info(f"Optimal F1 Score: {f1_score}, Accuracy: {accuracy}")

    return auc, aupr, f1_score, accuracy

# Save Best Model Function
def save_best_model(model, metrics, best_metrics, filename="best_model.pth"):
    """
    Save model if current metrics are better than best metrics.

    Args:
        model: PyTorch model to save
        metrics: Tuple of current metrics (AUC, AUPR, F1, ACC)
        best_metrics: Tuple of best metrics so far (AUC, AUPR, F1, ACC)
        filename: Path to save the model

    Returns:
        Updated best metrics
    """
    AUC, AUPR, F1, ACC = metrics
    best_AUC, best_AUPR, best_F1, best_ACC = best_metrics

    if AUC > best_AUC:
        best_AUC, best_AUPR, best_F1, best_ACC = AUC, AUPR, F1, ACC
        torch.save(model.state_dict(), filename)
        logging.info("✅ Best model saved!")

    return best_AUC, best_AUPR, best_F1, best_ACC
