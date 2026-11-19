"""Demo interactiva: sube una imagen o un video y la red estima si el rostro es falso.

Para cada rostro muestra la probabilidad de manipulación y el mapa Grad-CAM.
En los videos analiza varios frames y promedia, igual que en la evaluación.

Ejemplos:
    python -m app.demo --checkpoint runs/dos_ramas/mejor.pt
    python -m app.demo --checkpoint runs/dos_ramas/mejor.pt runs/meso4/mejor.pt --compartir   # en Colab
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.dataset import transformaciones
from src.data.extraer_rostros import cara_mas_grande, crear_detector, recortar
from src.gradcam import GradCAM, desnormalizar, superponer
from src.utils import cargar_modelo, elegir_dispositivo

AVISO = (
    "⚠️ Proyecto académico. La red se entrenó con FaceForensics++ y puede fallar con "
    "técnicas o videos muy distintos. Un resultado no es prueba de que un video sea real o falso."
)


class Detector:
    def __init__(self, checkpoints: list[Path]):
        self.dispositivo = elegir_dispositivo()
        self.modelos = {}
        for ck in checkpoints:
            modelo, tam = cargar_modelo(ck, self.dispositivo)
            nombre = f"{ck.parent.name} ({ck.name})" if ck.parent.name else ck.name
            self.modelos[nombre] = (modelo, tam, GradCAM(modelo))

    def rostro(self, frame_bgr: np.ndarray, tam: int) -> tuple[np.ndarray, bool]:
        """Recorta el rostro más grande. Si no hay rostro, usa la imagen completa."""
        alto, ancho = frame_bgr.shape[:2]
        caja = cara_mas_grande(crear_detector(ancho, alto), frame_bgr)
        if caja is None:
            return cv2.resize(frame_bgr, (tam, tam), interpolation=cv2.INTER_AREA), False
        return recortar(frame_bgr, caja, 1.3, tam), True

    def analizar(self, nombre: str, caras_bgr: list[np.ndarray]):
        modelo, tam, gc = self.modelos[nombre]
        tf = transformaciones(False, tam)
        x = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in caras_bgr])
        cams, probs = gc(x.to(self.dispositivo), objetivo="prediccion")
        superpuestas = [superponer(desnormalizar(xi), c.numpy()) for xi, c in zip(x, cams)]
        return probs.numpy(), superpuestas


def veredicto(p: float) -> str:
    if p >= 0.5:
        return f"### 🔴 Probablemente **manipulado** — p(falso) = {p:.2f}"
    return f"### 🟢 Probablemente **real** — p(falso) = {p:.2f}"


def frames_de_video(ruta: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(ruta)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1)), dtype=int).tolist())
    frames, i = [], 0
    while i <= max(indices):
        if i in indices:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return frames


def crear_interfaz(det: Detector) -> gr.Blocks:
    nombres = list(det.modelos)

    def imagen(img: np.ndarray, nombre: str):
        if img is None:
            return "Sube una imagen.", None, None
        _, tam, _ = det.modelos[nombre]
        cara, hallada = det.rostro(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), tam)
        probs, cams = det.analizar(nombre, [cara])
        texto = veredicto(float(probs[0]))
        if not hallada:
            texto += "\n\n_No se detectó un rostro: se analizó la imagen completa._"
        return texto, cv2.cvtColor(cara, cv2.COLOR_BGR2RGB), (cams[0] * 255).astype(np.uint8)

    def video(ruta: str, nombre: str, n_frames: int):
        if ruta is None:
            return "Sube un video.", None, None
        _, tam, _ = det.modelos[nombre]
        caras = [c for c, ok in (det.rostro(f, tam) for f in frames_de_video(ruta, int(n_frames))) if ok]
        if not caras:
            return "No se detectaron rostros en el video.", None, None
        probs, cams = det.analizar(nombre, caras)
        texto = veredicto(float(probs.mean())) + f"\n\nPromedio de {len(caras)} frames con rostro."
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.plot(probs, marker="o")
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylim(0, 1); ax.set_xlabel("frame analizado"); ax.set_ylabel("p(falso)")
        fig.tight_layout()
        galeria = [((c * 255).astype(np.uint8), f"p={p:.2f}") for c, p in zip(cams, probs)]
        return texto, fig, galeria

    with gr.Blocks(title="Detector de deepfakes") as demo:
        gr.Markdown("# Detector de rostros manipulados (deepfakes)\nRedes Neuronales y Aprendizaje Profundo — EAFIT")
        gr.Markdown(AVISO)
        modelo = gr.Dropdown(nombres, value=nombres[0], label="Modelo")
        with gr.Tab("Imagen"):
            with gr.Row():
                entrada = gr.Image(type="numpy", label="Imagen")
                with gr.Column():
                    resultado = gr.Markdown()
                    with gr.Row():
                        cara = gr.Image(label="Rostro detectado")
                        mapa = gr.Image(label="Grad-CAM")
            gr.Button("Analizar", variant="primary").click(imagen, [entrada, modelo], [resultado, cara, mapa])
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    entrada_v = gr.Video(label="Video")
                    n = gr.Slider(4, 32, value=16, step=1, label="Frames a analizar")
                with gr.Column():
                    resultado_v = gr.Markdown()
                    curva = gr.Plot(label="Probabilidad por frame")
            galeria = gr.Gallery(label="Grad-CAM por frame", columns=8, height="auto")
            gr.Button("Analizar video", variant="primary").click(
                video, [entrada_v, modelo, n], [resultado_v, curva, galeria])
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    p.add_argument("--compartir", action="store_true", help="Crea un enlace público temporal (útil en Colab)")
    p.add_argument("--puerto", type=int, default=7860)
    a = p.parse_args()
    crear_interfaz(Detector(a.checkpoint)).launch(share=a.compartir, server_port=a.puerto)


if __name__ == "__main__":
    main()
