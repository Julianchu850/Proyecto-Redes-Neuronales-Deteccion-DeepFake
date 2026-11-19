"""Registro de modelos. Todos reciben un tensor RGB normalizado (B, 3, H, W)
y devuelven un logit por imagen (B,) — positivo = falso."""

from functools import partial

from torch import nn

from .dos_ramas import DosRamas
from .efficientnet import EfficientNetB0
from .mesonet import Meso4

MODELOS = {
    # Línea base desde cero
    "meso4": Meso4,
    # Red propia y variantes para la ablación (E2)
    "dos_ramas": partial(DosRamas),
    "solo_espacial": partial(DosRamas, frecuencia=False),
    "solo_frecuencia": partial(DosRamas, espacial=False),
    "dos_ramas_sin_se": partial(DosRamas, se=False),
    "dos_ramas_srm": partial(DosRamas, entrada_frecuencia="srm"),
    # Referencia preentrenada
    "efficientnet_b0": EfficientNetB0,
}


def crear_modelo(nombre: str, tam_entrada: int = 224, **kwargs) -> nn.Module:
    if nombre not in MODELOS:
        raise ValueError(f"Modelo desconocido '{nombre}'. Opciones: {sorted(MODELOS)}")
    return MODELOS[nombre](tam_entrada=tam_entrada, **kwargs)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
