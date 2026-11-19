"""Registro de modelos. Todos reciben un tensor RGB normalizado (B, 3, H, W)
y devuelven un logit por imagen (B,) — positivo = falso."""

from torch import nn

from .mesonet import Meso4

MODELOS = {
    "meso4": Meso4,
    # Semana 2: "dos_ramas" (red propia) y "efficientnet_b0" (referencia preentrenada)
}


def crear_modelo(nombre: str, tam_entrada: int = 224) -> nn.Module:
    if nombre not in MODELOS:
        raise ValueError(f"Modelo desconocido '{nombre}'. Opciones: {sorted(MODELOS)}")
    return MODELOS[nombre](tam_entrada=tam_entrada)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
