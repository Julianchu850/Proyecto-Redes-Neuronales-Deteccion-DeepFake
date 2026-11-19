"""Extrae recortes de rostros de los videos de FaceForensics++ o Celeb-DF.

Por cada video toma N frames distribuidos uniformemente, detecta el rostro más
grande con YuNet (detector incluido en OpenCV), lo recorta con un margen
alrededor de la cara y lo guarda como PNG (sin pérdida, para no alterar las
huellas de frecuencia que buscamos detectar).

Al final escribe un manifiesto CSV con una fila por recorte.

Ejemplos:
    python -m src.data.extraer_rostros --dataset ffpp --raiz /ruta/FaceForensics --salida data/faces/ffpp
    python -m src.data.extraer_rostros --dataset celebdf --raiz /ruta/Celeb-DF-v2 --salida data/faces/celebdf
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PESOS_YUNET = Path(__file__).parent / "pesos" / "face_detection_yunet_2023mar.onnx"

METODOS_FFPP = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ---------------------------------------------------------------------------
# Listado de videos por dataset
# ---------------------------------------------------------------------------

def listar_videos_ffpp(raiz: Path, compresion: str) -> list[dict]:
    """Estructura que deja el script oficial de descarga de FaceForensics++."""
    videos = []
    reales = raiz / "original_sequences" / "youtube" / compresion / "videos"
    for v in sorted(reales.glob("*.mp4")):
        videos.append({"ruta": v, "etiqueta": 0, "metodo": "Original", "video_id": v.stem})
    for metodo in METODOS_FFPP:
        carpeta = raiz / "manipulated_sequences" / metodo / compresion / "videos"
        for v in sorted(carpeta.glob("*.mp4")):
            videos.append({"ruta": v, "etiqueta": 1, "metodo": metodo, "video_id": v.stem})
    return videos


def listar_videos_celebdf(raiz: Path) -> list[dict]:
    """Celeb-DF v2: Celeb-real/, YouTube-real/ y Celeb-synthesis/."""
    videos = []
    for carpeta, etiqueta in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        for v in sorted((raiz / carpeta).glob("*.mp4")):
            videos.append({"ruta": v, "etiqueta": etiqueta, "metodo": carpeta,
                           "video_id": f"{carpeta}/{v.stem}"})
    return videos


# ---------------------------------------------------------------------------
# Detección y recorte
# ---------------------------------------------------------------------------

def crear_detector(ancho: int, alto: int, umbral: float = 0.7):
    detector = cv2.FaceDetectorYN.create(str(PESOS_YUNET), "", (ancho, alto), umbral, 0.3, 5000)
    return detector


def cara_mas_grande(detector, frame: np.ndarray):
    """Devuelve (x, y, w, h) del rostro más grande o None."""
    _, caras = detector.detect(frame)
    if caras is None or len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])[:4]
    return float(x), float(y), float(w), float(h)


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


def procesar_video(info: dict, salida: Path, n_frames: int, margen: float, tam: int) -> list[dict]:
    cap = cv2.VideoCapture(str(info["ruta"]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return []

    indices = set(np.linspace(0, total - 1, num=min(n_frames, total), dtype=int).tolist())
    detector = crear_detector(ancho, alto)
    carpeta = salida / info["metodo"] / info["video_id"]
    carpeta.mkdir(parents=True, exist_ok=True)

    filas, ultima_caja, i = [], None, 0
    # Lectura secuencial: más confiable que saltar con CAP_PROP_POS_FRAMES en mp4
    while i <= max(indices):
        ok, frame = cap.read()
        if not ok:
            break
        if i in indices:
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
    return filas


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["ffpp", "celebdf"], required=True)
    p.add_argument("--raiz", type=Path, required=True, help="Carpeta raíz del dataset descargado")
    p.add_argument("--salida", type=Path, required=True, help="Carpeta donde se guardan los recortes")
    p.add_argument("--compresion", default="c23", help="Solo FF++: c23 o c40")
    p.add_argument("--frames-reales", type=int, default=32, help="Frames por video real")
    p.add_argument("--frames-falsos", type=int, default=8,
                   help="Frames por video falso (FF++ tiene 4 falsos por cada real; 32/8 balancea las clases)")
    p.add_argument("--margen", type=float, default=1.3)
    p.add_argument("--tam", type=int, default=224)
    p.add_argument("--procesos", type=int, default=4)
    p.add_argument("--limite", type=int, default=None, help="Procesar solo N videos (para pruebas)")
    a = p.parse_args()

    if a.dataset == "ffpp":
        videos = listar_videos_ffpp(a.raiz, a.compresion)
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
        w = csv.DictWriter(fh, fieldnames=["ruta", "etiqueta", "metodo", "video_id", "video", "frame"])
        w.writeheader()
        w.writerows(filas)
    n_videos = len({r["video"] for r in filas})
    print(f"Listo: {len(filas)} rostros de {n_videos} videos. Manifiesto: {manifiesto}")


if __name__ == "__main__":
    main()
