"""Entrena un modelo y guarda checkpoints reanudables.

Pensado para Colab: si la sesión se cae, se vuelve a correr el mismo comando
con --reanudar y continúa desde la última época guardada.

Ejemplo:
    python -m src.entrenar --modelo meso4 --datos data/faces/ffpp --splits data/splits/ffpp \
        --salida /content/drive/MyDrive/deepfakes/runs/meso4

Experimento E3 (técnica no vista): entrenar sin NeuralTextures
    python -m src.entrenar ... --metodos Deepfakes Face2Face FaceSwap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import RostrosDataset, transformaciones
from .metricas import calcular
from .models import contar_parametros, crear_modelo
from .utils import elegir_dispositivo, fijar_semilla, predecir


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True)
    p.add_argument("--datos", type=Path, required=True, help="Carpeta raíz de los recortes")
    p.add_argument("--splits", type=Path, required=True, help="Carpeta con train.csv y val.csv")
    p.add_argument("--salida", type=Path, required=True, help="Carpeta de checkpoints y registros")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tam", type=int, default=224)
    p.add_argument("--paciencia", type=int, default=5, help="Épocas sin mejorar el AUC de validación")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--metodos", nargs="+", default=None,
                   help="Técnicas de manipulación a usar en entrenamiento (por defecto, las 4 de FF++; FaceShifter nunca)")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--reanudar", action="store_true")
    return p.parse_args()


def main():
    a = argumentos()
    fijar_semilla(a.semilla)
    dispositivo = elegir_dispositivo()
    a.salida.mkdir(parents=True, exist_ok=True)

    ds_train = RostrosDataset(a.splits / "train.csv", a.datos, transformaciones(True, a.tam), a.metodos)
    ds_val = RostrosDataset(a.splits / "val.csv", a.datos, transformaciones(False, a.tam), a.metodos)
    cargar = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=dispositivo.type == "cuda",
                  persistent_workers=a.workers > 0)
    dl_train = DataLoader(ds_train, shuffle=True, drop_last=True, **cargar)
    dl_val = DataLoader(ds_val, shuffle=False, **cargar)

    modelo = crear_modelo(a.modelo, a.tam).to(dispositivo)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ds_train.peso_positivo(), device=dispositivo))
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.wd)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas)
    usar_amp = dispositivo.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)

    print(f"Dispositivo: {dispositivo} | modelo: {a.modelo} ({contar_parametros(modelo):,} parámetros)")
    print(f"Train: {len(ds_train)} rostros | Val: {len(ds_val)} rostros | pos_weight={ds_train.peso_positivo():.2f}")

    inicio, mejor_auc, sin_mejora, historial = 0, -1.0, 0, []
    ultimo = a.salida / "ultimo.pt"
    if a.reanudar and ultimo.exists():
        ck = torch.load(ultimo, map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        optimizador.load_state_dict(ck["optimizador"])
        planificador.load_state_dict(ck["planificador"])
        planificador.T_max = a.epocas  # por si se cambió --epocas al reanudar
        escalador.load_state_dict(ck["escalador"])
        inicio, mejor_auc, sin_mejora, historial = ck["epoca"] + 1, ck["mejor_auc"], ck["sin_mejora"], ck["historial"]
        print(f"Reanudando desde la época {inicio} (mejor AUC val = {mejor_auc:.4f})")
    else:
        (a.salida / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))

    for epoca in range(inicio, a.epocas):
        t0 = time.time()
        modelo.train()
        perdida_total, n = 0.0, 0
        for x, y, _ in tqdm(dl_train, desc=f"Época {epoca + 1}/{a.epocas}", leave=False):
            x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=usar_amp):
                perdida = criterio(modelo(x).float(), y)
            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()
            perdida_total += perdida.item() * len(y)
            n += len(y)
        planificador.step()

        val = calcular(predecir(modelo, dl_val, dispositivo))
        registro = {
            "epoca": epoca + 1,
            "perdida_train": perdida_total / max(n, 1),
            "auc_val_frame": val["frame"]["auc"],
            "auc_val_video": val["video"]["auc"],
            "acc_val_video": val["video"]["accuracy"],
            "lr": planificador.get_last_lr()[0],
            "segundos": round(time.time() - t0, 1),
        }
        historial.append(registro)
        pd.DataFrame(historial).to_csv(a.salida / "historial.csv", index=False)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in registro.items()))

        if registro["auc_val_frame"] > mejor_auc:
            mejor_auc, sin_mejora = registro["auc_val_frame"], 0
            torch.save({"modelo": modelo.state_dict(), "nombre": a.modelo, "tam": a.tam, "epoca": epoca + 1},
                       a.salida / "mejor.pt")
        else:
            sin_mejora += 1

        torch.save({"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
                    "planificador": planificador.state_dict(), "escalador": escalador.state_dict(),
                    "epoca": epoca, "mejor_auc": mejor_auc, "sin_mejora": sin_mejora,
                    "historial": historial}, ultimo)

        if sin_mejora >= a.paciencia:
            print(f"Early stopping: {a.paciencia} épocas sin mejorar.")
            break

    print(f"Mejor AUC de validación (frame): {mejor_auc:.4f} → {a.salida / 'mejor.pt'}")


if __name__ == "__main__":
    main()
