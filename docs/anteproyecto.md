# Anteproyecto: Detección de rostros manipulados (deepfakes) con redes neuronales

**Curso:** Redes Neuronales y Aprendizaje Profundo — Universidad EAFIT
**Integrantes:** Julián Lara Aristizábal · _[agregar integrantes]_
**Fecha:** octubre de 2026
**Duración estimada:** 3 semanas

---

## 1. Resumen

Este proyecto propone diseñar, entrenar y evaluar una red neuronal convolucional propia, construida desde cero, para clasificar rostros extraídos de videos como **reales** o **manipulados** (deepfakes). La arquitectura propuesta combina dos ramas: una **espacial**, que analiza la imagen del rostro, y una **de frecuencia**, que analiza su espectro de Fourier, donde los métodos de generación suelen dejar huellas invisibles al ojo humano.

El desempeño de la red se comparará con una línea base de la literatura (MesoNet, también entrenada desde cero) y con un modelo preentrenado de referencia (EfficientNet-B0). Además del rendimiento dentro del dataset de entrenamiento, el proyecto estudia el problema abierto más importante del área: **la generalización** a técnicas de manipulación y a datasets que el modelo no vio durante el entrenamiento.

## 2. Planteamiento del problema

Las herramientas de síntesis y manipulación de rostros con IA son cada vez más accesibles, y producen videos falsos difíciles de distinguir a simple vista. Esto facilita la desinformación, el fraude, la suplantación de identidad y el acoso.

Los detectores automáticos actuales alcanzan métricas muy altas cuando se evalúan con el mismo tipo de manipulación con el que se entrenaron, pero su rendimiento cae de forma marcada ante técnicas nuevas o videos de otras fuentes. Un detector útil en la práctica no puede depender de haber visto antes la técnica exacta del falsificador.

## 3. Justificación

- **Relevancia social:** la detección de contenido sintético es un problema vigente en seguridad, periodismo y plataformas digitales.
- **Interés técnico:** el proyecto permite trabajar diseño de arquitecturas, procesamiento de video, manejo de datos desbalanceados, prevención de fugas de información, ablaciones e interpretabilidad.
- **Viabilidad:** existen datasets académicos públicos y bien establecidos, y la arquitectura propuesta es lo bastante liviana para entrenarse con los recursos disponibles (sección 8).

## 4. Objetivos

### 4.1 Objetivo general

Diseñar y entrenar desde cero una red neuronal que clasifique rostros en video como reales o manipulados, y evaluar su capacidad de generalización frente a técnicas de manipulación y datasets no vistos durante el entrenamiento.

### 4.2 Objetivos específicos

1. Construir un pipeline de datos (extracción de frames, detección y recorte de rostros, división por video e identidad) libre de fugas de información.
2. Diseñar e implementar una arquitectura propia de dos ramas (espacial y de frecuencia).
3. Comparar la arquitectura propia con MesoNet (línea base) y EfficientNet-B0 preentrenado (referencia).
4. Cuantificar, mediante un estudio de ablación, el aporte de cada componente de la arquitectura.
5. Evaluar la generalización con técnicas de manipulación no vistas y con un dataset externo.
6. Interpretar las decisiones del modelo con mapas de activación (Grad-CAM).

## 5. Preguntas de investigación e hipótesis

- **P1.** ¿Puede una red pequeña entrenada desde cero acercarse al rendimiento de un modelo preentrenado mucho más grande?
  - _H1:_ Dentro del mismo dataset, la red propia alcanzará un AUC cercano al de EfficientNet-B0 con menos parámetros.
- **P2.** ¿Aporta la información del dominio de frecuencia a la detección?
  - _H2:_ La red de dos ramas superará a sus versiones de una sola rama, sobre todo al evaluar con técnicas de manipulación no vistas.
- **P3.** ¿Qué tanto se degrada el rendimiento al cambiar de dataset?
  - _H3:_ Todos los modelos perderán rendimiento en Celeb-DF; se espera que la rama de frecuencia reduzca esa pérdida.

## 6. Datos

| Dataset | Uso | Contenido | Acceso |
|---|---|---|---|
| **FaceForensics++** (versión comprimida c23) | Entrenamiento, validación y prueba | 1.000 videos reales y sus versiones manipuladas con Deepfakes, Face2Face, FaceSwap y NeuralTextures | Formulario de términos de uso → script de descarga ([repositorio oficial](https://github.com/ondyari/FaceForensics)) |
| **Celeb-DF v2** | Prueba externa (generalización) | Videos reales de celebridades y deepfakes de alta calidad | Formulario de solicitud ([repositorio oficial](https://github.com/yuezunli/celeb-deepfakeforensics)) |

**Plan de contingencia:** si el acceso a alguno de los datasets se retrasa, se iniciará el desarrollo del pipeline con subconjuntos públicos en Kaggle (por ejemplo, de DFDC) para no detener el cronograma.

**Consideraciones sobre los datos:**
- La división en entrenamiento, validación y prueba se hará **por video y por identidad**, nunca por frame, para evitar que la red memorice personas o escenas.
- FaceForensics++ tiene aproximadamente 4 videos manipulados por cada real; el desbalance se manejará con pérdida ponderada o muestreo balanceado.
- Los datos no se suben a este repositorio; solo se versionan los scripts y las listas de división.

## 7. Metodología

### 7.1 Preprocesamiento

1. Extracción de ~30 frames por video, distribuidos uniformemente.
2. Detección de rostro (MTCNN o RetinaFace) y recorte con un margen de ~1,3× alrededor de la cara.
3. Redimensionamiento a 224×224 px.
4. Aumento de datos: flips horizontales, variaciones de brillo y contraste, compresión JPEG y desenfoque aleatorios. Los dos últimos buscan que la red no aprenda a detectar "compresión" en lugar de "manipulación".

### 7.2 Modelos

| Modelo | Tipo | Rol en el proyecto |
|---|---|---|
| **MesoNet (Meso-4)** | Desde cero | Línea base de la literatura, diseñada para deepfakes |
| **Red de dos ramas (propuesta)** | Desde cero | Modelo principal del proyecto |
| **EfficientNet-B0** | Preentrenado (ImageNet) | Referencia de rendimiento ("techo") |

**Arquitectura propuesta (borrador):**

- **Rama espacial** (entrada RGB 3×224×224): un *stem* convolucional seguido de cuatro etapas de bloques residuales con convoluciones *depthwise separable* (64 → 128 → 256 → 384 canales), con atención *Squeeze-and-Excitation* en las dos últimas etapas, y *global average pooling*.
- **Rama de frecuencia** (entrada: logaritmo de la magnitud de la FFT 2D del rostro en escala de grises, centrada y estandarizada, 1×224×224): estructura análoga y más liviana (32 → 64 → 128 → 256 canales). Como variante se evaluarán filtros pasa-altos tipo SRM en lugar de la FFT.
- **Fusión:** concatenación de ambos vectores → *dropout* → capa densa → salida sigmoide.
- **Tamaño:** 1,63 millones de parámetros (frente a 4,0 millones de EfficientNet-B0).

La predicción a nivel de video se obtiene promediando las probabilidades de sus frames. La arquitectura final puede ajustarse según los resultados de las primeras pruebas.

### 7.3 Entrenamiento

Configuración inicial: optimizador AdamW, *learning rate* de 1e-3 con decaimiento coseno, *batch* de 64, entropía cruzada binaria ponderada, precisión mixta (AMP) y *early stopping* según el AUC de validación.

### 7.4 Experimentos

| ID | Experimento | Pregunta |
|---|---|---|
| E1 | Entrenamiento y prueba dentro de FaceForensics++ con los tres modelos | P1 |
| E2 | Ablación: solo rama espacial / solo frecuencia / ambas; con y sin SE | P2 |
| E3 | *Leave-one-manipulation-out*: entrenar con tres técnicas y probar con la cuarta | P2 |
| E4 | Generalización entre datasets: entrenar en FaceForensics++ y probar en Celeb-DF | P3 |
| E5 | Interpretabilidad con Grad-CAM sobre ejemplos reales y falsos | — |

### 7.5 Métricas

- **AUC-ROC** como métrica principal (estándar del área y robusta al desbalance).
- *Accuracy*, F1, precisión y *recall* como métricas complementarias.
- Todas se reportan a nivel de frame y a nivel de video.

## 8. Recursos

| Recurso | Uso previsto |
|---|---|
| Google Colab (GPU T4) | Entrenamiento principal |
| PC con GTX 1660 (6 GB) | Experimentos de ablación en paralelo |
| Mac con chip M5 (GPU integrada, backend MPS) | Preprocesamiento, evaluación, Grad-CAM y demo |

**Herramientas:** Python, PyTorch, OpenCV, facenet-pytorch o RetinaFace, scikit-learn, Matplotlib y Gradio.

## 9. Cronograma

| Semana | Actividades | Entregable parcial |
|---|---|---|
| 0 (inmediato) | Solicitar acceso a FaceForensics++ y Celeb-DF; configurar el repositorio | Anteproyecto |
| 1 | Pipeline de datos, análisis exploratorio, splits por video, MesoNet de punta a punta | Dataset procesado + línea base |
| 2 | Implementación de la red de dos ramas, E1, E2 y E3 | Resultados principales |
| 3 | E4, E5, demo en Gradio, informe y presentación | Informe final + demo |

## 10. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| Demora en el acceso a los datasets | Solicitud inmediata; plan B con datasets de Kaggle |
| Fuga de información entre *splits* | División por video e identidad, verificada con scripts |
| Sobreajuste a artefactos de compresión | Aumento de datos con compresión y evaluación en c23 y c40 |
| Desconexiones de Colab | Checkpoints en Google Drive en cada época |
| Tiempo de entrenamiento elevado | Resolución reducida (160–192 px) para ablaciones; 224 px solo para el modelo final |

## 11. Consideraciones éticas

Los datasets se usarán exclusivamente con fines académicos y según sus términos de uso. No se generarán deepfakes nuevos ni se usarán rostros de terceros fuera de los datasets. El informe incluirá una reflexión sobre las limitaciones del detector y sobre el riesgo de confiar ciegamente en sus predicciones.

## 12. Entregables

1. Repositorio con el código del pipeline, los modelos y los experimentos.
2. Informe técnico con resultados, ablaciones y análisis de generalización.
3. Demo interactiva en Gradio.
4. Presentación final.

## 13. Referencias

- Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a compact facial video forgery detection network. *IEEE WIFS*.
- Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. *CVPR*.
- Dolhansky, B., et al. (2020). The DeepFake Detection Challenge (DFDC) dataset. *arXiv:2006.07397*.
- Durall, R., Keuper, M., & Keuper, J. (2020). Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions. *CVPR*.
- Fridrich, J., & Kodovsky, J. (2012). Rich models for steganalysis of digital images. *IEEE Transactions on Information Forensics and Security*.
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR*.
- Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A large-scale challenging dataset for DeepFake forensics. *CVPR*.
- Qian, Y., Yin, G., Sheng, L., Chen, Z., & Shao, J. (2020). Thinking in frequency: Face forgery detection by mining frequency-aware clues. *ECCV*.
- Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. *ICCV*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
