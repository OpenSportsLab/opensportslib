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

def compute_classification_metrics(eval_pred, top_k=None, mode="logits"):
    """
    Compute accuracy, F1, precision, recall, and optionally top-k accuracy.
    Returns a dictionary for HF Trainer.
    """

    metrics = {}
    if mode=="labels":
        preds, labels = eval_pred
        preds = np.array(preds)
        labels = np.array(labels)
    else:
        preds, labels, topk_preds = process_preds_labels(eval_pred, top_k)

        # Top-k accuracy
        if top_k is not None and topk_preds is not None:
            topk_correct = sum([labels[i] in topk_preds[i] for i in range(len(labels))])
            metrics[f"top_{top_k}_accuracy"] = topk_correct / len(labels)

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


    return metrics

def compute_detailed_classification_metrics(all_logits, all_labels, class_names, save_dir, set_name):
    from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score

    preds = np.argmax(all_logits, axis=-1)

    sorted_class_names = sorted(class_names.values())
    name_to_sorted_idx = {name: i for i, name in enumerate(sorted_class_names)}
    idx_to_name = class_names

    sorted_labels = np.array([name_to_sorted_idx[idx_to_name[l]] for l in all_labels])
    sorted_preds = np.array([name_to_sorted_idx[idx_to_name[p]] for p in preds])

    all_class_labels = list(range(len(sorted_class_names)))

    cm = confusion_matrix(sorted_labels, sorted_preds, labels=all_class_labels)
    per_class_accuracy = np.diag(cm) / np.maximum(cm.sum(axis=1), 1) * 100
    balanced_acc = balanced_accuracy_score(sorted_labels, sorted_preds) * 100
    per_class_f1 = f1_score(sorted_labels, sorted_preds, labels=all_class_labels, average=None, zero_division=0) * 100
    macro_f1 = f1_score(sorted_labels, sorted_preds, labels=all_class_labels, average="macro", zero_division=0) * 100

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=sorted_class_names, yticklabels=sorted_class_names)
    plt.title(f'Confusion Matrix ({set_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{set_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    report_path = os.path.join(results_dir, f'{set_name}_detailed_metrics.txt')
    with open(report_path, 'w') as f:
        f.write(f"Balanced Accuracy: {balanced_acc:.2f}%\n")
        f.write(f"Macro F1:          {macro_f1:.2f}%\n\n")
        f.write(f"{'Class':<30} {'Accuracy':>10} {'F1':>10} {'Samples':>10}\n")
        f.write("-" * 65 + "\n")
        for i, class_name in enumerate(sorted_class_names):
            num_samples = int(cm[i].sum())
            f.write(f"{class_name:<30} {per_class_accuracy[i]:>9.2f}% {per_class_f1[i]:>9.2f}% {num_samples:>10}\n")
        f.write("-" * 65 + "\n\n")
        f.write("Classification Report:\n\n")
        f.write(classification_report(
            sorted_labels, sorted_preds,
            labels=all_class_labels,
            target_names=sorted_class_names,
            zero_division=0
        ))
        f.write("\n" + "-" * 65 + "\n\n")
        f.write("Confusion Matrix:\n\n")
        f.write(f"{cm}\n")
    
    tsv_path = os.path.join(results_dir, f'{set_name}_results.tsv')
    with open(tsv_path, 'w') as f:
        header = "metric\t" + "\t".join(sorted_class_names) + "\toverall"
        f.write(header + "\n")

        acc_row = "accuracy\t" + "\t".join(f"{per_class_accuracy[i]:.2f}" for i in range(len(sorted_class_names))) + f"\t{balanced_acc:.2f}"
        f.write(acc_row + "\n")

        f1_row = "f1\t" + "\t".join(f"{per_class_f1[i]:.2f}" for i in range(len(sorted_class_names))) + f"\t{macro_f1:.2f}"
        f.write(f1_row + "\n")

        samples_row = "samples\t" + "\t".join(str(int(cm[i].sum())) for i in range(len(sorted_class_names))) + f"\t{int(cm.sum())}"
        f.write(samples_row + "\n")

    print(f"Saved TSV to {tsv_path}")
    
    print(f"\nSaved detailed metrics to {report_path}")
    print(f"\nBalanced Accuracy: {balanced_acc:.2f}%")
    print(f"Macro F1:          {macro_f1:.2f}%\n")
    print(f"{'Class':<30} {'Accuracy':>10} {'F1':>10} {'Samples':>10}")
    print("-" * 65)
    for i, class_name in enumerate(sorted_class_names):
        num_samples = int(cm[i].sum())
        print(f"{class_name:<30} {per_class_accuracy[i]:>9.2f}% {per_class_f1[i]:>9.2f}% {num_samples:>10}")
    print("-" * 65)
    
    return {
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "per_class_accuracy": {name: per_class_accuracy[i] for i, name in enumerate(sorted_class_names)},
        "per_class_f1": {name: per_class_f1[i] for i, name in enumerate(sorted_class_names)},
    }
    