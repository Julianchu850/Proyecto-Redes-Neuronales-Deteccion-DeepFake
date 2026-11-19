# Detección de deepfakes con redes neuronales

Proyecto del curso **Redes Neuronales y Aprendizaje Profundo** (Universidad EAFIT).

Diseñamos y entrenamos desde cero una red neuronal convolucional de dos ramas, **espacial** y **de frecuencia**, que clasifica rostros extraídos de videos como reales o manipulados. Luego la comparamos con MesoNet y con EfficientNet-B0 preentrenado, y estudiamos qué tan bien generaliza a técnicas de manipulación y datasets que no vio durante el entrenamiento.

- 📄 [Propuesta corta](docs/propuesta.md)
- 📄 [Anteproyecto completo](docs/anteproyecto.md): problema, objetivos, hipótesis, metodología, cronograma y riesgos.

## Estado

| Semana | Contenido | Estado |
|---|---|---|
| 1 | Pipeline de datos (extracción de rostros, splits sin fugas), MesoNet, entrenamiento y evaluación | ✅ Código listo — falta correr con los datos reales |
| 2 | Red propia de dos ramas, EfficientNet-B0, ablaciones (E1–E3) | ⏳ |
| 3 | Celeb-DF (E4), Grad-CAM (E5), demo e informe | ⏳ |

## Estructura del repositorio

```
├── data/
│   └── splits/ffpp_oficial/   # División oficial de FaceForensics++ por pares de identidades
├── docs/                      # Propuesta, anteproyecto e informe
├── notebooks/
│   └── 01_entrenamiento_colab.ipynb
├── results/                   # Métricas de los experimentos
└── src/
    ├── data/
    │   ├── extraer_rostros.py  # Videos → recortes de rostros (detector YuNet) + manifiesto
    │   ├── crear_splits.py     # Manifiesto → train/val/test sin fugas de identidad
    │   └── dataset.py          # Dataset de PyTorch y aumentos (JPEG, desenfoque, color)
    ├── models/
    │   └── mesonet.py          # Meso-4, línea base desde cero
    ├── entrenar.py             # Entrenamiento con AMP, early stopping y checkpoints reanudables
    ├── evaluar.py              # Métricas por frame, por video y por técnica
    ├── metricas.py
    └── utils.py
```

## Uso

### 1. Instalación

```bash
pip install -r requirements.txt
```

`src/utils.py` elige el dispositivo automáticamente: CUDA (Colab, GTX 1660) → MPS (Mac) → CPU.

### 2. Datos

Ver [`data/README.md`](data/README.md) para descargar FaceForensics++ y Celeb-DF.

### 3. Extraer rostros

```bash
python -m src.data.extraer_rostros --dataset ffpp --raiz /ruta/FaceForensics --salida data/faces/ffpp
```

Por defecto toma 32 frames de cada video real y 8 de cada falso. Como FaceForensics++ tiene 4 videos falsos por cada real, eso deja las clases balanceadas (~32 mil rostros de cada una). Los recortes se guardan en PNG para no agregar compresión que confunda a la rama de frecuencia.

### 4. Crear los splits

```bash
python -m src.data.crear_splits --dataset ffpp --rostros data/faces/ffpp --salida data/splits/ffpp
```

Usa la división oficial de FaceForensics++ y verifica que ninguna identidad quede en dos particiones.

### 5. Entrenar y evaluar

```bash
python -m src.entrenar --modelo meso4 --datos data/faces/ffpp --splits data/splits/ffpp --salida runs/meso4
python -m src.evaluar --checkpoint runs/meso4/mejor.pt --datos data/faces/ffpp \
    --csv data/splits/ffpp/test.csv --salida results/meso4_ffpp
```

En Colab, usar [`notebooks/01_entrenamiento_colab.ipynb`](notebooks/01_entrenamiento_colab.ipynb).

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics): entrenamiento, validación y prueba.
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics): prueba de generalización.

Ambos requieren solicitar acceso y aceptar sus términos de uso. Los datos **no** se suben a este repositorio.

## Créditos de terceros

- División oficial de FaceForensics++ (`data/splits/ffpp_oficial/`), tomada del [repositorio de FaceForensics](https://github.com/ondyari/FaceForensics).
- Detector de rostros YuNet (`src/data/pesos/`), del [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (licencia MIT).

## Integrantes

- Julián Lara Aristizábal
- _[agregar integrantes]_
