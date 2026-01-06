import numpy as np
from evaluate import load
from sklearn.metrics import balanced_accuracy_score

# Load HuggingFace metrics
accuracy_metric = load("accuracy")
f1_metric = load("f1")
precision_metric = load("precision")
recall_metric = load("recall")

def process_preds_labels(eval_pred, top_k=None):
    """
    Handles tuple logits, one-hot labels, and optionally returns top-k predictions.
    """
    logits, labels = eval_pred

    # Handle tuple outputs (some HF models return tuple)
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # Convert one-hot labels to class indices
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)

    # # Ensure logits are 2D for top-k computations
    # if logits.ndim == 1:
    #     logits = logits.reshape(1, -1)

    # Predicted classes
    preds = np.argmax(logits, axis=-1)

    # Top-k predictions for top-k accuracy
    if top_k is not None:
        topk_preds = np.argsort(logits, axis=-1)[:, -top_k:]
    else:
        topk_preds = None

    return preds, labels, topk_preds

def compute_classification_metrics(eval_pred, top_k=None):
    """
    Compute accuracy, F1, precision, recall, and optionally top-k accuracy.
    Returns a dictionary for HF Trainer.
    """
    preds, labels, topk_preds = process_preds_labels(eval_pred, top_k)

    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    # Balanced accuracy
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, preds)

    # F1 (macro)
    metrics["f1"] = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    # Precision
    metrics["precision"] = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]

    # Recall
    metrics["recall"] = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]

    # Top-k accuracy
    if top_k is not None and topk_preds is not None:
        topk_correct = sum([labels[i] in topk_preds[i] for i in range(len(labels))])
        metrics[f"top_{top_k}_accuracy"] = topk_correct / len(labels)

    return metrics
