"""Métricas a nivel de frame y de video."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _metricas(y: np.ndarray, prob: np.ndarray, umbral: float = 0.5) -> dict:
    pred = (prob >= umbral).astype(int)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "n": int(len(y)),
    }


def por_video(df: pd.DataFrame) -> pd.DataFrame:
    """Promedia la probabilidad de los frames de cada video."""
    return df.groupby("video", as_index=False).agg(
        prob=("prob", "mean"), etiqueta=("etiqueta", "first"), metodo=("metodo", "first")
    )


def calcular(df: pd.DataFrame) -> dict:
    """df con columnas: prob, etiqueta, video, metodo."""
    vid = por_video(df)
    res = {
        "frame": _metricas(df["etiqueta"].to_numpy(), df["prob"].to_numpy()),
        "video": _metricas(vid["etiqueta"].to_numpy(), vid["prob"].to_numpy()),
    }
    # AUC por técnica de manipulación: reales vs. falsos de esa técnica
    metodos_falsos = sorted(df.loc[df["etiqueta"] == 1, "metodo"].unique())
    if len(metodos_falsos) > 1:
        res["por_metodo"] = {}
        for m in metodos_falsos:
            sub = df[(df["etiqueta"] == 0) | (df["metodo"] == m)]
            sub_vid = por_video(sub)
            res["por_metodo"][m] = {
                "auc_frame": _metricas(sub["etiqueta"].to_numpy(), sub["prob"].to_numpy())["auc"],
                "auc_video": _metricas(sub_vid["etiqueta"].to_numpy(), sub_vid["prob"].to_numpy())["auc"],
            }
    return res
