"""Balanced linear probe for gold position from frozen hidden states.

Consumes the JSONL that ``probe_scan`` writes and fits a logistic
regression that discriminates gold-at-an-edge from gold-at-the-middle
using only the frozen hidden-state vector. A shuffled-label control fits
the same probe on permuted labels; a probe that recovers position from
real labels but not shuffled labels shows the location is linearly
decodable from the representation.

The classifier is a plain gradient-descent logistic regression written
against numpy so the module has no scikit-learn dependency and stays on
the pinned runtime. Evaluation is grouped k-fold by question so a
question's edge and middle vectors never straddle the train/test split.
"""

import hashlib
from collections.abc import Iterable

EDGE_POSITIONS = (0, 1, 8, 9)
MIDDLE_POSITIONS = (4, 5)
DEFAULT_FOLDS = 5
DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.1
PROBE_SEED = 20260130


def _label_for(position: int) -> int | None:
    if position in EDGE_POSITIONS:
        return 1
    if position in MIDDLE_POSITIONS:
        return 0
    return None


def _stable_fold(question_id: str, folds: int) -> int:
    digest = hashlib.sha256(question_id.encode()).hexdigest()
    return int(digest, 16) % folds


def _standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def _fit_logistic(x, y, epochs: int, lr: float):
    import numpy as np

    n_features = x.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    n = x.shape[0]
    for _ in range(epochs):
        logits = x @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-logits))
        error = preds - y
        weights -= lr * (x.T @ error) / n
        bias -= lr * error.mean()
    return weights, bias


def _predict(x, weights, bias):
    import numpy as np

    logits = x @ weights + bias
    return (1.0 / (1.0 + np.exp(-logits))) >= 0.5


def _balance_indices(labels, rng):
    import numpy as np

    labels = np.asarray(labels)
    positive = list(np.where(labels == 1)[0])
    negative = list(np.where(labels == 0)[0])
    k = min(len(positive), len(negative))
    if k == 0:
        return []
    rng.shuffle(positive)
    rng.shuffle(negative)
    keep = positive[:k] + negative[:k]
    rng.shuffle(keep)
    return keep


def probe_gold_position(
    records: Iterable[dict],
    folds: int = DEFAULT_FOLDS,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    seed: int = PROBE_SEED,
) -> dict:
    """Fit and cross-validate a balanced edge-vs-middle position probe.

    Returns::

        {
            "layer": int,
            "n_samples": int,
            "n_features": int,
            "accuracy": float,        # grouped k-fold, balanced classes
            "shuffled_accuracy": float,
            "folds": int,
            "per_fold_accuracy": [float, ...],
        }

    ``accuracy`` well above ``shuffled_accuracy`` (which should hover at
    the 0.5 chance line) means gold position is linearly decodable from
    the chosen layer's frozen hidden state.
    """
    import numpy as np

    records = list(records)
    features = []
    labels = []
    groups = []
    layer = None
    for record in records:
        label = _label_for(int(record["gold_position"]))
        if label is None:
            continue
        features.append(record["hidden_state"])
        labels.append(label)
        groups.append(record["question_id"])
        if layer is None:
            layer = int(record["layer"])
    if not features:
        raise ValueError("no edge/middle records to probe")

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    def run(label_vector) -> tuple[float, list[float]]:
        fold_accuracies = []
        for fold in range(folds):
            test_mask = np.array([_stable_fold(qid, folds) == fold for qid in groups])
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            train_idx = _balance_indices(label_vector[train_mask], rng)
            if not train_idx:
                continue
            x_train_all = features[train_mask]
            y_train_all = label_vector[train_mask]
            x_train = x_train_all[train_idx]
            y_train = y_train_all[train_idx]
            x_test = features[test_mask]
            y_test = label_vector[test_mask]
            test_idx = _balance_indices(y_test, rng)
            if not test_idx:
                continue
            x_test = x_test[test_idx]
            y_test = y_test[test_idx]
            x_train_std, x_test_std = _standardize(x_train, x_test)
            weights, bias = _fit_logistic(x_train_std, y_train, epochs, lr)
            predictions = _predict(x_test_std, weights, bias)
            fold_accuracies.append(float((predictions == (y_test >= 0.5)).mean()))
        overall = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
        return overall, fold_accuracies

    accuracy, per_fold = run(labels)
    shuffled_labels = labels.copy()
    rng.shuffle(shuffled_labels)
    shuffled_accuracy, _ = run(shuffled_labels)

    return {
        "layer": layer,
        "n_samples": len(labels),
        "n_features": int(features.shape[1]),
        "accuracy": accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "folds": folds,
        "per_fold_accuracy": per_fold,
    }
