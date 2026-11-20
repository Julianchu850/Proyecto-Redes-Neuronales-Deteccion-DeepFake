"""Utilidades compartidas: dispositivo, semillas y predicción."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def elegir_dispositivo() -> torch.device:
    """CUDA (Kaggle / GTX 1660) → MPS (Mac) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def cargar_modelo(checkpoint, dispositivo: torch.device):
    """Carga un checkpoint 'mejor.pt' y devuelve (modelo en modo eval, tamaño de entrada)."""
    from .models import crear_modelo  # import local para evitar ciclos

    ck = torch.load(checkpoint, map_location=dispositivo, weights_only=False)
    extra = {"preentrenado": False} if ck["nombre"] == "efficientnet_b0" else {}  # los pesos vienen del checkpoint
    modelo = crear_modelo(ck["nombre"], ck["tam"], **extra).to(dispositivo)
    modelo.load_state_dict(ck["modelo"])
    modelo.eval()
    return modelo, ck["tam"]


@torch.no_grad()
def predecir(modelo, loader: DataLoader, dispositivo: torch.device) -> pd.DataFrame:
    """Devuelve un DataFrame con las filas del dataset y la probabilidad de 'falso'."""
    modelo.eval()
    probs, indices = [], []
    usar_amp = dispositivo.type == "cuda"
    for x, _, idx in loader:
        x = x.to(dispositivo, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=usar_amp):
            logits = modelo(x)
        probs.append(torch.sigmoid(logits.float()).cpu())
        indices.append(idx)
    df = loader.dataset.df.iloc[torch.cat(indices).numpy()].copy()
    df["prob"] = torch.cat(probs).numpy()
    return df
