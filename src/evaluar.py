"""Evalúa un checkpoint en un split y guarda métricas y predicciones.

Ejemplos:
    # E1: prueba dentro de FaceForensics++
    python -m src.evaluar --checkpoint runs/meso4/mejor.pt --datos data/faces/ffpp \
        --csv data/splits/ffpp/test.csv --salida results/meso4_ffpp

    # E4: generalización a Celeb-DF
    python -m src.evaluar --checkpoint runs/meso4/mejor.pt --datos data/faces/celebdf \
        --csv data/splits/celebdf/test.csv --salida results/meso4_celebdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import RostrosDataset, transformaciones
from .metricas import calcular
from .models import crear_modelo
from .utils import elegir_dispositivo, predecir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--datos", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    dispositivo = elegir_dispositivo()
    ck = torch.load(a.checkpoint, map_location=dispositivo, weights_only=False)
    extra = {"preentrenado": False} if ck["nombre"] == "efficientnet_b0" else {}  # los pesos vienen del checkpoint
    modelo = crear_modelo(ck["nombre"], ck["tam"], **extra).to(dispositivo)
    modelo.load_state_dict(ck["modelo"])

    ds = RostrosDataset(a.csv, a.datos, transformaciones(False, ck["tam"]))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    pred = predecir(modelo, dl, dispositivo)
    res = calcular(pred)

    a.salida.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.salida / "predicciones.csv", index=False)
    (a.salida / "metricas.json").write_text(json.dumps(res, indent=2))

    print(f"Frame → AUC {res['frame']['auc']:.4f} | acc {res['frame']['accuracy']:.4f} | F1 {res['frame']['f1']:.4f}")
    print(f"Video → AUC {res['video']['auc']:.4f} | acc {res['video']['accuracy']:.4f} | F1 {res['video']['f1']:.4f}")
    for m, v in res.get("por_metodo", {}).items():
        print(f"  {m:15s} AUC frame {v['auc_frame']:.4f} | AUC video {v['auc_video']:.4f}")
    print(f"Resultados en {a.salida}")


if __name__ == "__main__":
    main()
