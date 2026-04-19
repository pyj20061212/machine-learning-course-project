import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_logits = []
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_logits.append(logits.cpu().numpy())
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_logits, all_preds, all_labels


def compute_classification_metrics(y_true, y_pred, class_names=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }



@torch.no_grad()
def collect_predictions_with_probs(model, loader, device):
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_images = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_images.append(images.cpu().numpy())

    return {
        "logits": np.concatenate(all_logits, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "preds": np.concatenate(all_preds, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "images": np.concatenate(all_images, axis=0),
    }


@torch.no_grad()
def extract_feature_vectors(model, loader, device, feature_key: str = "pool2"):
    model.eval()

    all_features = []
    all_labels = []
    all_images = []

    for images, labels in loader:
        images = images.to(device)
        feats = model.forward_features(images)
        x = feats[feature_key]
        x = x.view(x.size(0), -1)

        all_features.append(x.cpu().numpy())
        all_labels.append(labels.numpy())
        all_images.append(images.cpu().numpy())

    return {
        "features": np.concatenate(all_features, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "images": np.concatenate(all_images, axis=0),
    }


def find_top_confident_examples(probs, labels, target_class: int, top_k: int = 8):
    class_mask = labels == target_class
    class_indices = np.where(class_mask)[0]
    class_scores = probs[class_indices, target_class]
    order = np.argsort(-class_scores)
    return class_indices[order[:top_k]]


def find_most_confused_pairs(confusion_matrix, top_k: int = 5):
    cm = confusion_matrix.copy().astype(np.float64)
    np.fill_diagonal(cm, 0.0)

    pairs = []
    flat_idx = np.argsort(cm.ravel())[::-1]
    used = 0
    for idx in flat_idx:
        i, j = np.unravel_index(idx, cm.shape)
        if cm[i, j] <= 0:
            continue
        pairs.append((i, j, int(cm[i, j])))
        used += 1
        if used >= top_k:
            break
    return pairs