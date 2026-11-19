"""Extrae recortes de rostros de los videos de FaceForensics++ o Celeb-DF.

Por cada video toma N frames distribuidos uniformemente, detecta el rostro más
grande con YuNet (detector incluido en OpenCV), lo recorta con un margen
alrededor de la cara y lo guarda como PNG (sin pérdida, para no alterar las
huellas de frecuencia que buscamos detectar).

Al final escribe un manifiesto CSV con una fila por recorte. Si se interrumpe,
al volver a correr el mismo comando salta los videos ya procesados.

Ejemplos:
    python -m src.data.extraer_rostros --dataset ffpp --raiz /ruta/FaceForensics --salida data/faces/ffpp
    python -m src.data.extraer_rostros --dataset celebdf --raiz /ruta/Celeb-DF-v2 --salida data/faces/celebdf
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PESOS_YUNET = Path(__file__).parent / "pesos" / "face_detection_yunet_2023mar.onnx"

METODOS_FFPP = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
# FaceShifter viene en algunas versiones del dataset: se extrae aparte y se usa
# solo como técnica nunca vista en entrenamiento (prueba de generalización).
METODOS_EXTRA = ["FaceShifter"]
NOMBRES_ORIGINAL = {"original", "original_sequences", "youtube", "real", "originals"}
ID_FFPP = re.compile(r"^\d{3}(_\d{3})?$")  # 000 (original) o 000_003 (manipulado)


# ---------------------------------------------------------------------------
# Listado de videos por dataset
# ---------------------------------------------------------------------------

def listar_videos_ffpp(raiz: Path, compresion: str, metodos: list[str]) -> list[dict]:
    """Busca los .mp4 bajo `raiz` y deduce la técnica por el nombre de las carpetas.

    Funciona con la estructura del script oficial
        original_sequences/youtube/c23/videos/000.mp4
        manipulated_sequences/Deepfakes/c23/videos/000_003.mp4
    y con versiones reorganizadas como la de Kaggle
        FaceForensics++_C23/original/000.mp4
        FaceForensics++_C23/Deepfakes/000_003.mp4
    DeepFakeDetection (actores pagados, otros ids) se ignora.
    """
    por_nombre = {m.lower(): m for m in metodos}
    otras_compresiones = {"raw", "c23", "c40"} - {compresion}
    videos, ignorados = [], 0
    for v in sorted(raiz.rglob("*.mp4")):
        partes = [p.lower() for p in v.relative_to(raiz).parts[:-1]]
        if any(p in otras_compresiones for p in partes) or not ID_FFPP.match(v.stem):
            ignorados += 1
            continue
        metodo = next((por_nombre[p] for p in reversed(partes) if p in por_nombre), None)
        if metodo is not None and "_" in v.stem:
            videos.append({"ruta": v, "etiqueta": 1, "metodo": metodo, "video_id": v.stem})
        elif metodo is None and "_" not in v.stem and any(p in NOMBRES_ORIGINAL for p in partes):
            videos.append({"ruta": v, "etiqueta": 0, "metodo": "Original", "video_id": v.stem})
        else:
            ignorados += 1
    conteo = {}
    for v in videos:
        conteo[v["metodo"]] = conteo.get(v["metodo"], 0) + 1
    print("Videos encontrados:", ", ".join(f"{k}={n}" for k, n in sorted(conteo.items())),
          f"| ignorados: {ignorados}")
    return videos


def listar_videos_celebdf(raiz: Path) -> list[dict]:
    """Celeb-DF v2: busca las carpetas Celeb-real/, YouTube-real/ y Celeb-synthesis/ bajo `raiz`."""
    videos = []
    for carpeta, etiqueta in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        encontradas = [d for d in raiz.rglob(carpeta) if d.is_dir()]
        for d in encontradas[:1]:
            for v in sorted(d.glob("*.mp4")):
                videos.append({"ruta": v, "etiqueta": etiqueta, "metodo": carpeta,
                               "video_id": f"{carpeta}/{v.stem}"})
    return videos


# ---------------------------------------------------------------------------
# Detección y recorte
# ---------------------------------------------------------------------------

# La cara se busca en una copia reducida del frame (más rápido en videos HD);
# el recorte sí se hace sobre el frame original, sin perder calidad.
MAX_LADO_DETECCION = 640


def crear_detector(ancho: int, alto: int, umbral: float = 0.7):
    escala = min(1.0, MAX_LADO_DETECCION / max(ancho, alto, 1))
    tam = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    return cv2.FaceDetectorYN.create(str(PESOS_YUNET), "", tam, umbral, 0.3, 5000)


def cara_mas_grande(detector, frame: np.ndarray):
    """Devuelve (x, y, w, h) del rostro más grande, en coordenadas del frame original, o None."""
    w_in, h_in = detector.getInputSize()
    alto, ancho = frame.shape[:2]
    if (ancho, alto) != (w_in, h_in):
        frame = cv2.resize(frame, (w_in, h_in), interpolation=cv2.INTER_AREA)
    _, caras = detector.detect(frame)
    if caras is None or len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])[:4]
    sx, sy = ancho / w_in, alto / h_in
    return float(x) * sx, float(y) * sy, float(w) * sx, float(h) * sy


def recortar(frame: np.ndarray, caja, margen: float, tam: int) -> np.ndarray:
    """Recorte cuadrado centrado en la cara, ampliado por `margen`."""
    x, y, w, h = caja
    cx, cy = x + w / 2, y + h / 2
    lado = max(w, h) * margen
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x1, y1 = int(round(cx + lado / 2)), int(round(cy + lado / 2))
    # Relleno con bordes negros si el recorte se sale de la imagen
    H, W = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    cara = frame[y0:y1, x0:x1]
    return cv2.resize(cara, (tam, tam), interpolation=cv2.INTER_AREA)


CAMPOS = ["ruta", "etiqueta", "metodo", "video_id", "video", "frame"]


def procesar_video(info: dict, salida: Path, n_frames: int, margen: float, tam: int) -> list[dict]:
    carpeta = salida / info["metodo"] / info["video_id"]
    listo = carpeta / "_listo.csv"
    if listo.exists():  # ya procesado en una ejecución anterior: permite reanudar
        with open(listo, newline="") as fh:
            return list(csv.DictReader(fh))

    cap = cv2.VideoCapture(str(info["ruta"]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return []

    indices = set(np.linspace(0, total - 1, num=min(n_frames, total), dtype=int).tolist())
    detector = crear_detector(ancho, alto)
    carpeta.mkdir(parents=True, exist_ok=True)

    filas, ultima_caja, i = [], None, 0
    # Lectura secuencial: más confiable que saltar con CAP_PROP_POS_FRAMES en mp4
    while i <= max(indices):
        if i not in indices:
            if not cap.grab():  # avanza sin convertir el frame: más rápido que read()
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        caja = cara_mas_grande(detector, frame)
        if caja is None:
            caja = ultima_caja  # si no detecta, reutiliza la última caja conocida
        if caja is not None:
            ultima_caja = caja
            ruta = carpeta / f"{i:04d}.png"
            cv2.imwrite(str(ruta), recortar(frame, caja, margen, tam))
            filas.append({
                "ruta": str(ruta.relative_to(salida)),
                "etiqueta": info["etiqueta"],
                "metodo": info["metodo"],
                "video_id": info["video_id"],
                "video": f'{info["metodo"]}/{info["video_id"]}',
                "frame": i,
            })
        i += 1
    cap.release()
    with open(listo, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    return filas


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["ffpp", "celebdf"], required=True)
    p.add_argument("--raiz", type=Path, required=True, help="Carpeta raíz del dataset descargado")
    p.add_argument("--salida", type=Path, required=True, help="Carpeta donde se guardan los recortes")
    p.add_argument("--compresion", default="c23", help="Solo FF++: c23 o c40")
    p.add_argument("--sin-faceshifter", action="store_true",
                   help="Solo FF++: no extraer FaceShifter aunque esté en el dataset")
    p.add_argument("--frames-reales", type=int, default=32, help="Frames por video real")
    p.add_argument("--frames-falsos", type=int, default=8,
                   help="Frames por video falso (FF++ tiene 4 falsos por cada real; 32/8 balancea las clases)")
    p.add_argument("--margen", type=float, default=1.3)
    p.add_argument("--tam", type=int, default=224)
    p.add_argument("--procesos", type=int, default=4)
    p.add_argument("--limite", type=int, default=None, help="Procesar solo N videos (para pruebas)")
    a = p.parse_args()

    if a.dataset == "ffpp":
        metodos = METODOS_FFPP + ([] if a.sin_faceshifter else METODOS_EXTRA)
        videos = listar_videos_ffpp(a.raiz, a.compresion, metodos)
    else:
        videos = listar_videos_celebdf(a.raiz)
    if a.limite:
        videos = videos[: a.limite]
    if not videos:
        raise SystemExit(f"No se encontraron videos en {a.raiz}. Revisa la estructura de carpetas.")

    a.salida.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} videos → {a.salida}")

    filas = []
    with ProcessPoolExecutor(max_workers=a.procesos) as ex:
        futuros = [
            ex.submit(procesar_video, v, a.salida,
                      a.frames_reales if v["etiqueta"] == 0 else a.frames_falsos, a.margen, a.tam)
            for v in videos
        ]
        for f in tqdm(as_completed(futuros), total=len(futuros)):
            filas.extend(f.result())

    filas.sort(key=lambda r: r["ruta"])
    manifiesto = a.salida / "manifiesto.csv"
    with open(manifiesto, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    n_videos = len({r["video"] for r in filas})
    print(f"Listo: {len(filas)} rostros de {n_videos} videos. Manifiesto: {manifiesto}")


if __name__ == "__main__":
    main()
