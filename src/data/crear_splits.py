"""Divide el manifiesto de rostros en train / val / test SIN fugas de información.

FaceForensics++: usa la división oficial del dataset (data/splits/ffpp_oficial/*.json),
que agrupa los videos por pares de identidades. Un video manipulado "A_B" va a la
misma partición que los originales A y B, así ninguna persona aparece en dos
particiones. El script verifica que no haya cruces.

Celeb-DF v2: solo se usa como prueba externa, con la lista oficial
List_of_testing_videos.txt.

Ejemplos:
    python -m src.data.crear_splits --dataset ffpp --rostros data/faces/ffpp --salida data/splits/ffpp
    python -m src.data.crear_splits --dataset celebdf --rostros data/faces/celebdf \
        --lista-celebdf /ruta/Celeb-DF-v2/List_of_testing_videos.txt --salida data/splits/celebdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

OFICIAL_FFPP = Path(__file__).resolve().parents[2] / "data" / "splits" / "ffpp_oficial"


def particion_ffpp(video_id: str, ids_por_split: dict[str, set[str]]) -> str | None:
    """'071' → split del original; '071_054' → split del par (ambos ids deben coincidir)."""
    ids = video_id.split("_")
    for split, conjunto in ids_por_split.items():
        if all(i in conjunto for i in ids):
            return split
    return None


def splits_ffpp(manifiesto: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ids_por_split = {}
    for split in ["train", "val", "test"]:
        pares = json.loads((OFICIAL_FFPP / f"{split}.json").read_text())
        ids_por_split[split] = {i for par in pares for i in par}

    manifiesto = manifiesto.copy()
    manifiesto["split"] = manifiesto["video_id"].astype(str).map(lambda v: particion_ffpp(v, ids_por_split))
    sin_split = manifiesto["split"].isna().sum()
    if sin_split:
        print(f"Aviso: {sin_split} recortes sin partición asignada (se descartan)")
    manifiesto = manifiesto.dropna(subset=["split"])

    # Verificación de fugas: ninguna identidad en dos particiones
    identidades = {
        s: {i for v in df["video_id"].astype(str) for i in v.split("_")}
        for s, df in manifiesto.groupby("split")
    }
    for a in identidades:
        for b in identidades:
            if a < b:
                cruce = identidades[a] & identidades[b]
                assert not cruce, f"Fuga de identidades entre {a} y {b}: {sorted(cruce)[:5]}"
    print("Verificación OK: ninguna identidad aparece en dos particiones.")
    return {s: df.drop(columns="split") for s, df in manifiesto.groupby("split")}


def splits_celebdf(manifiesto: pd.DataFrame, lista: Path) -> dict[str, pd.DataFrame]:
    prueba = set()
    for linea in lista.read_text().splitlines():
        if linea.strip():
            _, ruta = linea.split()
            prueba.add(ruta.removesuffix(".mp4"))
    return {"test": manifiesto[manifiesto["video_id"].isin(prueba)]}


def resumen(nombre: str, df: pd.DataFrame) -> None:
    reales = (df["etiqueta"] == 0).sum()
    falsos = (df["etiqueta"] == 1).sum()
    print(f"  {nombre:5s}: {len(df):7d} rostros | {df['video'].nunique():5d} videos | "
          f"reales {reales} / falsos {falsos}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["ffpp", "celebdf"], required=True)
    p.add_argument("--rostros", type=Path, required=True, help="Carpeta con manifiesto.csv")
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--lista-celebdf", type=Path, help="List_of_testing_videos.txt (solo Celeb-DF)")
    a = p.parse_args()

    manifiesto = pd.read_csv(a.rostros / "manifiesto.csv", dtype={"video_id": str})
    if a.dataset == "ffpp":
        splits = splits_ffpp(manifiesto)
    else:
        if not a.lista_celebdf:
            raise SystemExit("Para Celeb-DF indica --lista-celebdf (List_of_testing_videos.txt, viene con el dataset)")
        splits = splits_celebdf(manifiesto, a.lista_celebdf)

    a.salida.mkdir(parents=True, exist_ok=True)
    for nombre, df in splits.items():
        df.to_csv(a.salida / f"{nombre}.csv", index=False)
        resumen(nombre, df)
    print(f"Splits guardados en {a.salida}")


if __name__ == "__main__":
    main()
