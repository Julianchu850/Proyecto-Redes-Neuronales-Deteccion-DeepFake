"""Red propia de dos ramas (espacial + frecuencia), diseñada y entrenada desde cero.

Idea: los deepfakes dejan dos tipos de huellas.
- Espaciales: bordes de la cara mal fusionados, color inconsistente, ojos o dientes raros.
- De frecuencia: el upsampling de los generadores deja patrones periódicos de alta
  frecuencia, casi invisibles en la imagen pero claros en su espectro.

La red recibe la misma imagen RGB que los otros modelos y calcula internamente la
entrada de la rama de frecuencia, así que entrenamiento, evaluación y Grad-CAM son
iguales para todos los modelos.

Variantes para el estudio de ablación (ver MODELOS en __init__.py):
    dos_ramas          espacial + frecuencia (FFT), con SE
    solo_espacial      solo la rama espacial
    solo_frecuencia    solo la rama de frecuencia
    dos_ramas_sin_se   sin atención Squeeze-and-Excitation
    dos_ramas_srm      frecuencia con filtros pasa-altos SRM en lugar de FFT
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Bloques
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Atención por canal (Hu et al., 2018): aprende qué mapas de características pesar más."""

    def __init__(self, canales: int, reduccion: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(canales, canales // reduccion, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales // reduccion, canales, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


def conv_separable(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    """Convolución depthwise 3×3 + pointwise 1×1 (Chollet, 2017): menos parámetros que una 3×3 normal."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),
        nn.BatchNorm2d(c_in),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_in, c_out, 1, bias=False),
        nn.BatchNorm2d(c_out),
    )


class BloqueResidual(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, se: bool = False):
        super().__init__()
        self.cuerpo = nn.Sequential(
            conv_separable(c_in, c_out, stride),
            nn.ReLU(inplace=True),
            conv_separable(c_out, c_out),
        )
        self.se = SqueezeExcitation(c_out) if se else nn.Identity()
        self.atajo = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.atajo = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.se(self.cuerpo(x)) + self.atajo(x))


class Rama(nn.Module):
    """Stem + 4 etapas (cada una reduce la resolución a la mitad) + global average pooling."""

    def __init__(self, c_in: int, stem: int, canales: list[int], se_en: set[int], bloques_por_etapa: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, stem, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
        )
        etapas, previo = [], stem
        for i, c in enumerate(canales):
            bloques = [BloqueResidual(previo, c, stride=2, se=i in se_en)]
            bloques += [BloqueResidual(c, c, se=i in se_en) for _ in range(bloques_por_etapa - 1)]
            etapas.append(nn.Sequential(*bloques))
            previo = c
        self.etapas = nn.Sequential(*etapas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim_salida = previo

    def forward(self, x):
        return self.pool(self.etapas(self.stem(x))).flatten(1)


# ---------------------------------------------------------------------------
# Entradas de la rama de frecuencia
# ---------------------------------------------------------------------------

def a_grises(x: torch.Tensor) -> torch.Tensor:
    """Deshace la normalización de ImageNet y pasa a escala de grises: (B,3,H,W) → (B,1,H,W) en [0,1]."""
    x = x * DESV.to(x.device) + MEDIA.to(x.device)
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0, 1)


def espectro_fft(x: torch.Tensor) -> torch.Tensor:
    """log(|FFT2D| + ε), centrado y estandarizado por imagen.

    Se usa log(|F| + ε) y no log(1 + |F|): las magnitudes de alta frecuencia son
    mucho menores que 1, y log1p las dejaría casi en cero, justo donde están los
    artefactos que buscamos.
    """
    g = a_grises(x)
    mag = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1)).abs()
    esp = torch.log(mag + 1e-6)
    media = esp.mean(dim=(-2, -1), keepdim=True)
    desv = esp.std(dim=(-2, -1), keepdim=True) + 1e-6
    return (esp - media) / desv


class FiltrosSRM(nn.Module):
    """Tres filtros pasa-altos fijos de los "rich models" de esteganálisis (Fridrich y Kodovsky, 2012).

    Eliminan el contenido de la imagen y dejan el "ruido" residual, donde suelen
    quedar los artefactos de manipulación.
    """

    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
                           [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 4
        k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12
        k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
                           [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 2
        self.register_buffer("kernels", torch.stack([k1, k2, k3]).unsqueeze(1))  # (3,1,5,5), no se entrenan

    def forward(self, x):
        r = F.conv2d(a_grises(x), self.kernels, padding=2)
        return r / (r.std(dim=(-2, -1), keepdim=True) + 1e-6)


# ---------------------------------------------------------------------------
# Red completa
# ---------------------------------------------------------------------------

class DosRamas(nn.Module):
    def __init__(self, tam_entrada: int = 224, espacial: bool = True, frecuencia: bool = True,
                 se: bool = True, entrada_frecuencia: str = "fft", dropout: float = 0.5):
        super().__init__()
        assert espacial or frecuencia, "Se necesita al menos una rama"
        assert entrada_frecuencia in ("fft", "srm")
        se_en = {2, 3} if se else set()  # SE en las dos últimas etapas

        self.espacial = Rama(3, 32, [64, 128, 256, 384], se_en) if espacial else None
        self.frecuencia = None
        if frecuencia:
            self.entrada_frecuencia = entrada_frecuencia
            self.srm = FiltrosSRM() if entrada_frecuencia == "srm" else None
            c_in = 3 if entrada_frecuencia == "srm" else 1
            self.frecuencia = Rama(c_in, 32, [32, 64, 128, 256], se_en)

        dim = (self.espacial.dim_salida if espacial else 0) + (self.frecuencia.dim_salida if frecuencia else 0)
        self.clasificador = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def entrada_rama_frecuencia(self, x: torch.Tensor) -> torch.Tensor:
        # La FFT en media precisión falla con tamaños que no son potencia de 2 y pierde
        # precisión, así que esta parte siempre se calcula en float32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            return self.srm(x) if self.srm is not None else espectro_fft(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rasgos = []
        if self.espacial is not None:
            rasgos.append(self.espacial(x))
        if self.frecuencia is not None:
            rasgos.append(self.frecuencia(self.entrada_rama_frecuencia(x)))
        return self.clasificador(torch.cat(rasgos, dim=1)).squeeze(1)

    def capa_gradcam(self) -> nn.Module:
        """Última etapa de la rama espacial (mapa de 7×7), para Grad-CAM."""
        rama = self.espacial if self.espacial is not None else self.frecuencia
        return rama.etapas[-1]
