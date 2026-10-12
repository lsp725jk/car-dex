# -*- coding: utf-8 -*-
"""评估指标：51 类 Macro-F1（唯一评分指标）+ 辅助分析指标。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def fmt_rate(v: float | None, digits: int = 4) -> str:
    """把可能为 None 的比率格式化成字符串（None → `—`）。

    微调时用的自定义小数据集里常常没有 unknown 样本，
    这时 unknown 检出率是"无定义"的 —— 显示成 `—` 比显示 `nan` 清楚得多，
    也不会让下游的 round()/格式化直接崩掉。
    """
    return "—" if v is None else f"{v:.{digits}f}"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    """按任务书 4.3 节口径计算指标。

    评分指标：51 类 Macro-F1（等权平均，分母为 0 的类别 F1 记 0）。
    辅助指标：Accuracy、每类 P/R/F1、混淆矩阵、unknown 检出率、已知类误拒率。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(label_names)
    unknown_index = n_classes - 1

    labels = list(range(n_classes))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # unknown 检出率：真实 unknown 中被正确判为 unknown 的比例。
    # ⚠️ 用自定义小数据集微调时，验证集里**可能一个 unknown 样本都没有** ——
    # 那种情况下这个指标是"无定义"，返回 None 让调用方显示成 `—`，
    # 而不是 NaN（NaN 会污染日志、也会让下游的排序/比较莫名其妙地失效）。
    unk_true = y_true == unknown_index
    unknown_detection_rate = (
        float((y_pred[unk_true] == unknown_index).mean()) if unk_true.any() else None
    )
    # 已知类误拒率：真实已知类中被误判为 unknown 的比例
    known_true = ~unk_true
    known_false_reject_rate = (
        float((y_pred[known_true] == unknown_index).mean()) if known_true.any() else None
    )

    per_class = []
    for i, name in enumerate(label_names):
        per_class.append(
            {
                "index": i,
                "label": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "unknown_detection_rate": unknown_detection_rate,
        "known_false_reject_rate": known_false_reject_rate,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "report_text": classification_report(
            y_true, y_pred, labels=labels, target_names=label_names, zero_division=0, digits=3
        ),
        "n_samples": int(len(y_true)),
    }


def top_confusions(cm: np.ndarray, label_names: list[str], top_k: int = 25) -> list[dict]:
    """从混淆矩阵中提取最严重的类别混淆对（排除对角线）。"""
    cm = np.asarray(cm)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append(
                {
                    "true": label_names[i],
                    "pred": label_names[j],
                    "count": int(cm[i, j]),
                    "true_total": int(cm[i].sum()),
                    "rate": float(cm[i, j] / max(1, cm[i].sum())),
                }
            )
    pairs.sort(key=lambda d: (-d["count"], -d["rate"]))
    return pairs[:top_k]
