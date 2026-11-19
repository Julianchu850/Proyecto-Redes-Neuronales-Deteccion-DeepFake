"""MesoNet (Meso-4) — línea base del proyecto, implementada desde cero.

Afchar et al. (2018), "MesoNet: a compact facial video forgery detection network".
Red pequeña (~24 mil parámetros con entrada de 224 px) que se enfoca en propiedades "mesoscópicas"
de la imagen: ni el ruido de píxel ni la semántica de alto nivel.

Diferencia con el artículo: la salida es un logit (sin sigmoide) para usar
BCEWithLogitsLoss, que es numéricamente más estable.
"""

import torch
from torch import nn


def bloque(c_in: int, c_out: int, k: int, pool: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class Meso4(nn.Module):
    def __init__(self, tam_entrada: int = 224):
        super().__init__()
        self.extractor = nn.Sequential(
            bloque(3, 8, 3, 2),
            bloque(8, 8, 5, 2),
            bloque(8, 16, 5, 2),
            bloque(16, 16, 5, 4),
        )
        with torch.no_grad():
            n = self.extractor(torch.zeros(1, 3, tam_entrada, tam_entrada)).numel()
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(n, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.extractor(x)).squeeze(1)
