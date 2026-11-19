"""Dataset de PyTorch y aumentos de datos para los recortes de rostros."""

from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T

# Técnicas que nunca se usan para entrenar: solo prueban la generalización
METODOS_RESERVADOS = ["FaceShifter"]

MEDIA = (0.485, 0.456, 0.406)
DESV = (0.229, 0.224, 0.225)


class CompresionJPEG:
    """Recomprime la imagen como JPEG con calidad aleatoria.

    Obliga a la red a no confundir "imagen comprimida" con "imagen manipulada".
    """

    def __init__(self, calidad=(60, 100), p=0.5):
        self.calidad, self.p = calidad, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.calidad))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class DesenfoqueGaussiano:
    def __init__(self, radio=(0.1, 1.5), p=0.2):
        self.radio, self.p = radio, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radio)))


def transformaciones(entrenamiento: bool, tam: int = 224) -> T.Compose:
    if entrenamiento:
        return T.Compose([
            T.Resize((tam, tam)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            CompresionJPEG(),
            DesenfoqueGaussiano(),
            T.ToTensor(),
            T.Normalize(MEDIA, DESV),
        ])
    return T.Compose([T.Resize((tam, tam)), T.ToTensor(), T.Normalize(MEDIA, DESV)])


class RostrosDataset(Dataset):
    """Lee un CSV de split (ruta, etiqueta, metodo, video_id, video, frame).

    video_id identifica a las personas (para dividir sin fugas); video identifica
    cada video de forma única (técnica + id), para agrupar predicciones.

    Devuelve (imagen, etiqueta, índice de fila) — el índice permite agrupar
    las predicciones por video al evaluar.
    """

    def __init__(self, csv: Path, raiz: Path, transform=None, metodos: list[str] | None = None):
        """metodos: técnicas de manipulación a conservar (los reales siempre se conservan).

        Por defecto se usan todas excepto las reservadas para generalización
        (FaceShifter), que solo entran si se piden explícitamente.
        """
        df = pd.read_csv(csv, dtype={"video_id": str})
        falsos = df["etiqueta"] == 1
        if metodos is not None:
            df = df[~falsos | df["metodo"].isin(metodos)]
        else:
            df = df[~falsos | ~df["metodo"].isin(METODOS_RESERVADOS)]
        self.df = df.reset_index(drop=True)
        self.raiz = Path(raiz)
        self.transform = transform or transformaciones(False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        fila = self.df.iloc[i]
        img = Image.open(self.raiz / fila["ruta"]).convert("RGB")
        return self.transform(img), torch.tensor(float(fila["etiqueta"])), i

    def peso_positivo(self) -> float:
        """reales / falsos, para BCEWithLogitsLoss(pos_weight=...)."""
        falsos = (self.df["etiqueta"] == 1).sum()
        reales = (self.df["etiqueta"] == 0).sum()
        return float(reales / max(falsos, 1))
