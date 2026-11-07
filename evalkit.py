import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve


PATCH_PX = 512 * 512


def confusion(y_true, y_pred, weight=None):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    w = np.ones(len(y_true)) if weight is None else np.asarray(weight, dtype=np.float64)
    return {"TP": float(w[y_true & y_pred].sum()), "FP": float(w[~y_true & y_pred].sum()),
            "FN": float(w[y_true & ~y_pred].sum()), "TN": float(w[~y_true & ~y_pred].sum())}


def scores(cm):
    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    div = lambda a, b: a / b if b > 0 else 0.0
    prev = div(TP + FN, TP + FP + FN + TN)
    return {"precision": div(TP, TP + FP), "recall": div(TP, TP + FN),
            "f1": div(2 * TP, 2 * TP + FP + FN), "iou": div(TP, TP + FP + FN),
            "fpr": div(FP, FP + TN), "prevalence": prev, "flag_all_f1": div(2 * prev, 1 + prev)}


def best_f1_threshold(y_true, score, weight=None):
    pr, rc, th = precision_recall_curve(y_true, score, sample_weight=weight)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(th[np.argmax(f1[:-1])])


def average_precision(y_true, score, weight=None):
    return float(average_precision_score(y_true, score, sample_weight=weight))


def plot_confusion(cm, title, ax):
    M = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    R = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{R[i, j]:.1%}\n({M[i, j]:,.0f})", ha="center", va="center",
                    color="white" if R[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred no fire", "pred fire"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true no fire", "true fire"])
    ax.set_title(title, fontsize=10)

def recall_at_fpr(y_true, score, fpr_max, weight=None):
    fpr, tpr, _ = roc_curve(y_true, score, sample_weight=weight)
    return float(tpr[fpr <= fpr_max].max())