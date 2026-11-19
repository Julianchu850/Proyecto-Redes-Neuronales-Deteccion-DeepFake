# Propuesta de proyecto: Detección de deepfakes con redes neuronales

**Curso:** Redes Neuronales y Aprendizaje Profundo — EAFIT
**Integrantes:** Julián Lara Aristizábal · _[agregar integrantes]_

**Problema.** Los videos con rostros manipulados por IA (deepfakes) son cada vez más fáciles de producir y difíciles de detectar a simple vista, lo que facilita la desinformación y la suplantación de identidad. Los detectores actuales funcionan bien con las técnicas que vieron en su entrenamiento, pero fallan ante técnicas nuevas.

**Objetivo.** Diseñar y entrenar desde cero una red neuronal convolucional que clasifique rostros extraídos de videos como reales o manipulados, y evaluar qué tan bien generaliza a técnicas y datasets no vistos durante el entrenamiento.

**Propuesta de solución.** Una CNN propia de dos ramas:

- Una **rama espacial**, que analiza la imagen del rostro.
- Una **rama de frecuencia**, que analiza su espectro de Fourier, donde los métodos de generación dejan huellas invisibles al ojo.

Su rendimiento se comparará con MesoNet, una red pequeña de la literatura entrenada también desde cero, y con EfficientNet-B0 preentrenado como referencia.

**Datos.** Dos datasets académicos públicos:

- **FaceForensics++:** 1.000 videos reales y sus versiones manipuladas con cuatro técnicas. Se usará para entrenamiento y prueba.
- **Celeb-DF v2:** se usará como prueba externa de generalización.

**Evaluación.**

- La métrica principal será el AUC-ROC.
- Se hará un estudio de ablación para medir el aporte de cada rama.
- Se probará la red con una técnica de manipulación que no vio en el entrenamiento y con un dataset externo.
- Se interpretarán sus decisiones con Grad-CAM.

**Recursos y tiempo.** El proyecto toma 3 semanas y se trabajará con Google Colab (GPU T4) y equipos propios.

**Entregables.**

- Repositorio con el código.
- Informe técnico.
- Demo interactiva.

El planteamiento detallado está en [`anteproyecto.md`](anteproyecto.md).
