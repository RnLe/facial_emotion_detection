"""Scores computed from predicted probabilities and true labels (torch tensors)."""
import torch


def confusion(pred, true, classes=7):
    return torch.bincount(true * classes + pred, minlength=classes * classes).reshape(classes, classes)


def scores(prob, true, classes=7, bins=15):
    """Accuracy, macro-F1, balanced accuracy, per-class recall and precision, the confusion
    matrix, and the expected calibration error (how far confidence is from accuracy)."""
    pred = prob.argmax(1)
    cm = confusion(pred, true, classes).double()
    tp = cm.diag()
    recall = tp / cm.sum(1).clamp(min=1)
    precision = tp / cm.sum(0).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    conf = prob.max(1).values
    correct = (pred == true).double()
    edges = torch.linspace(0, 1, bins + 1, device=prob.device)
    ece = torch.zeros((), device=prob.device, dtype=torch.double)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.double().mean() * (conf[m].double().mean() - correct[m].mean()).abs()

    return {
        "accuracy": correct.mean().item(),
        "macro_f1": f1.mean().item(),
        "balanced_accuracy": recall.mean().item(),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "confusion": cm.long().tolist(),
        "ece": ece.item(),
    }
