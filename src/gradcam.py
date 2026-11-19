"""Grad-CAM (Selvaraju et al., 2017): ¿en qué parte del rostro se fija la red? (experimento E5)

Genera dos figuras a partir de un checkpoint y un split:
1. Ejemplos: rostros reales y de cada técnica, con el mapa de calor encima.
2. Mapa promedio por técnica: promedia el Grad-CAM de muchos rostros. Como los
   recortes están centrados en la cara, el promedio muestra qué regiones usa la
   red en general (bordes de la cara, boca, ojos...) para cada técnica.

En la red de dos ramas se usa la rama espacial, que es la que tiene
correspondencia directa con los píxeles del rostro.

Ejemplo:
    python -m src.gradcam --checkpoint runs/dos_ramas/mejor.pt --datos data/faces/ffpp \
        --csv data/splits/ffpp/test.csv --salida results/gradcam/dos_ramas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import DESV, MEDIA, RostrosDataset, transformaciones
from .utils import cargar_modelo, elegir_dispositivo


class GradCAM:
    def __init__(self, modelo: torch.nn.Module, capa: torch.nn.Module | None = None):
        self.modelo = modelo
        capa = capa if capa is not None else modelo.capa_gradcam()
        self._act = self._grad = None
        capa.register_forward_hook(self._guardar)

    def _guardar(self, modulo, entrada, salida):
        self._act = salida
        if salida.requires_grad:
            salida.register_hook(lambda g: setattr(self, "_grad", g))

    def __call__(self, x: torch.Tensor, objetivo: str = "prediccion"):
        """Devuelve (mapas (B,H,W) en [0,1], probabilidad de 'falso' (B,)).

        objetivo='prediccion': explica la clase que predijo la red.
        objetivo='falso':      explica la evidencia a favor de 'falso'.
        """
        self.modelo.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # asegura que haya grafo aunque la capa esté al inicio
            logits = self.modelo(x)
            signo = torch.ones_like(logits)
            if objetivo == "prediccion":
                signo = torch.where(logits >= 0, 1.0, -1.0)
            self.modelo.zero_grad(set_to_none=True)
            (logits * signo).sum().backward()

        pesos = self._grad.mean(dim=(2, 3), keepdim=True)          # importancia de cada canal
        cam = F.relu((pesos * self._act).sum(dim=1, keepdim=True))  # (B,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        maximo = cam.flatten(1).max(dim=1).values.view(-1, 1, 1)
        cam = cam / (maximo + 1e-8)
        return cam.detach().cpu(), torch.sigmoid(logits.detach()).cpu()


def desnormalizar(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalizado → imagen (H,W,3) en [0,1]."""
    m = torch.tensor(MEDIA).view(3, 1, 1)
    s = torch.tensor(DESV).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def superponer(img: np.ndarray, cam: np.ndarray, alfa: float = 0.45) -> np.ndarray:
    calor = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alfa) * img + alfa * calor, 0, 1)


def un_frame_por_video(df, semilla: int):
    return df.groupby("video", group_keys=False).sample(1, random_state=semilla)


def figura_ejemplos(gc, ds, dispositivo, por_tecnica: int, semilla: int, ruta: Path):
    frames = un_frame_por_video(ds.df, semilla)
    tecnicas = ["Original"] + sorted(t for t in frames["metodo"].unique() if t != "Original")
    fig, ax = plt.subplots(2 * len(tecnicas), por_tecnica,
                           figsize=(2.1 * por_tecnica, 2.2 * 2 * len(tecnicas)), squeeze=False)
    for fila, tecnica in enumerate(tecnicas):
        sub = frames[frames["metodo"] == tecnica]
        sub = sub.sample(min(por_tecnica, len(sub)), random_state=semilla)
        x = torch.stack([ds[i][0] for i in sub.index]).to(dispositivo)
        cams, probs = gc(x)
        for j in range(por_tecnica):
            a_img, a_cam = ax[2 * fila, j], ax[2 * fila + 1, j]
            a_img.axis("off"); a_cam.axis("off")
            if j >= len(sub):
                continue
            img = desnormalizar(x[j])
            a_img.imshow(img)
            a_cam.imshow(superponer(img, cams[j].numpy()))
            pred = "falso" if probs[j] >= 0.5 else "real"
            correcto = pred == ("real" if tecnica == "Original" else "falso")
            a_img.set_title(f"{tecnica}\np(falso)={probs[j]:.2f} {'✓' if correcto else '✗'}",
                            fontsize=8, color="black" if correcto else "red")
    fig.suptitle("Grad-CAM: regiones que explican la predicción", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def figura_promedio(gc, ds, dispositivo, n: int, semilla: int, lote: int, ruta: Path):
    frames = un_frame_por_video(ds.df, semilla)
    tecnicas = ["Original"] + sorted(t for t in frames["metodo"].unique() if t != "Original")
    fig, ax = plt.subplots(1, len(tecnicas), figsize=(2.6 * len(tecnicas), 3), squeeze=False)
    for k, tecnica in enumerate(tecnicas):
        sub = frames[frames["metodo"] == tecnica]
        sub = sub.sample(min(n, len(sub)), random_state=semilla)
        suma_cam, suma_img = 0, 0
        for inicio in range(0, len(sub), lote):
            idx = sub.index[inicio:inicio + lote]
            x = torch.stack([ds[i][0] for i in idx]).to(dispositivo)
            cams, _ = gc(x, objetivo="falso")
            suma_cam = suma_cam + cams.sum(0)
            suma_img = suma_img + sum(desnormalizar(xi) for xi in x)
        cam = (suma_cam / len(sub)).numpy()
        cam = cam / (cam.max() + 1e-8)
        ax[0, k].imshow(superponer(np.clip(suma_img / len(sub), 0, 1), cam, alfa=0.55))
        ax[0, k].set_title(f"{tecnica}\n(n={len(sub)})", fontsize=9)
        ax[0, k].axis("off")
    fig.suptitle("Grad-CAM promedio (evidencia de 'falso') por técnica", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--datos", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--por-tecnica", type=int, default=6, help="Ejemplos por técnica en la figura de ejemplos")
    p.add_argument("--n-promedio", type=int, default=200, help="Rostros por técnica para el mapa promedio")
    p.add_argument("--metodos", nargs="+", default=None)
    p.add_argument("--semilla", type=int, default=0)
    a = p.parse_args()

    dispositivo = elegir_dispositivo()
    modelo, tam = cargar_modelo(a.checkpoint, dispositivo)
    gc = GradCAM(modelo)
    ds = RostrosDataset(a.csv, a.datos, transformaciones(False, tam), a.metodos)

    a.salida.mkdir(parents=True, exist_ok=True)
    figura_ejemplos(gc, ds, dispositivo, a.por_tecnica, a.semilla, a.salida / "gradcam_ejemplos.png")
    figura_promedio(gc, ds, dispositivo, a.n_promedio, a.semilla, 32, a.salida / "gradcam_promedio.png")
    print(f"Figuras en {a.salida}")


if __name__ == "__main__":
    main()
