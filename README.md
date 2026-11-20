# Detección de deepfakes con redes neuronales

Proyecto del curso **Redes Neuronales y Aprendizaje Profundo** (Universidad EAFIT).

Diseñamos y entrenamos desde cero una red neuronal convolucional de dos ramas, **espacial** y **de frecuencia**, que clasifica rostros extraídos de videos como reales o manipulados. Luego la comparamos con MesoNet y con EfficientNet-B0 preentrenado, y estudiamos qué tan bien generaliza a técnicas de manipulación y datasets que no vio durante el entrenamiento.

## Resultados

Todos los modelos se entrenaron con la misma receta en una GPU T4 de Kaggle y se evaluaron en el conjunto de prueba de FaceForensics++ (C23): 140 videos originales y sus versiones manipuladas, de personas que ninguna red vio. La métrica es el AUC por video (promedio de las caras de cada video). Las cifras completas están en [`results/resultados.csv`](results/resultados.csv).

**Hallazgos principales:**

1. Desde cero, la red propia (0,893) supera a MesoNet (0,828). EfficientNet-B0 preentrenado sigue arriba (0,984).
2. Lo que gana en lo conocido no es lo que generaliza: la rama de frecuencia sola es la peor en FaceForensics++ (0,700), pero ante FaceShifter, que ningún modelo vio, llega a 0,709, igual que EfficientNet.
3. Ante una técnica no vista, la red propia cae de ~0,92 a entre 0,43 y 0,78. FaceSwap queda por debajo del azar (0,425).
4. La variante con filtros SRM es la mejor desde cero (0,939); la atención SE no aportó.

Cada modelo se entrenó una sola vez, así que las diferencias pequeñas pueden ser ruido.

### E1: prueba en FaceForensics++

| Modelo | Parámetros | AUC frame | AUC video | Accuracy video |
|---|---|---|---|---|
| MesoNet (desde cero) | 24 mil | 0,788 | 0,828 | 72,1 % |
| Red propia (desde cero) | 1,63 M | 0,854 | 0,893 | 83,3 % |
| EfficientNet-B0 (preentrenado) | 4,0 M | 0,979 | 0,984 | 96,3 % |

| AUC video por técnica | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|
| MesoNet | 0,951 | 0,856 | 0,784 | 0,723 |
| Red propia | 0,942 | 0,903 | 0,935 | 0,792 |
| EfficientNet-B0 | 0,996 | 0,986 | 0,989 | 0,964 |

### E2: ablación y E3b: FaceShifter

| Variante | Qué cambia | AUC video FF++ | AUC video FaceShifter |
|---|---|---|---|
| Red completa | Espacial + frecuencia (FFT), con SE | 0,893 | 0,640 |
| Solo espacial | Sin rama de frecuencia | 0,922 | 0,582 |
| Solo frecuencia | Sin rama espacial | 0,700 | 0,709 |
| Sin SE | Sin atención por canal | 0,899 | 0,651 |
| Con SRM | Filtros pasa-altos en vez de FFT | 0,939 | 0,655 |
| MesoNet | Línea base | 0,828 | 0,652 |
| EfficientNet-B0 | Preentrenado | 0,984 | 0,709 |

### E3: técnica no vista (red propia)

Cada fila es una red entrenada sin una técnica. «Vistas» es el promedio de las otras tres.

| Técnica no vista | AUC en la no vista | AUC en las vistas |
|---|---|---|
| Deepfakes | 0,781 | 0,892 |
| Face2Face | 0,673 | 0,902 |
| FaceSwap | 0,425 | 0,944 |
| NeuralTextures | 0,520 | 0,938 |

### E5: Grad-CAM

Promedio de 140 caras de prueba por técnica. La red propia (arriba, rama espacial) mira una zona amplia del centro de la cara, casi igual en todas las técnicas. EfficientNet-B0 (abajo) señala un lugar distinto por técnica; en NeuralTextures, la boca, que es lo único que esa técnica modifica.

![Grad-CAM promedio de la red propia](docs/figuras/gradcam_promedio_dos_ramas.png)

![Grad-CAM promedio de EfficientNet-B0](docs/figuras/gradcam_promedio_efficientnet_b0.png)

El experimento E4 (generalización a Celeb-DF v2) no se alcanzó a correr y queda como trabajo futuro.

## Estructura del repositorio

```
├── app/
│   └── demo.py                # Demo en Gradio: imagen o video → probabilidad + Grad-CAM
├── data/
│   └── splits/ffpp_oficial/   # División oficial de FaceForensics++ por pares de identidades
├── docs/                      # Propuesta, anteproyecto e informe
├── notebooks/
│   ├── kaggle_01_extraer_rostros.ipynb  # Kaggle: rostros + splits
│   └── kaggle_02_entrenar.ipynb         # Kaggle: todos los experimentos
├── results/
│   └── resultados.csv         # Métricas de todos los experimentos
└── src/
    ├── data/
    │   ├── extraer_rostros.py  # Videos → recortes de rostros (detector YuNet) + manifiesto
    │   ├── crear_splits.py     # Manifiesto → train/val/test sin fugas de identidad
    │   └── dataset.py          # Dataset de PyTorch y aumentos (JPEG, desenfoque, color)
    ├── models/
    │   ├── mesonet.py          # Meso-4, línea base desde cero
    │   ├── dos_ramas.py        # Red propia: rama espacial + rama de frecuencia (FFT o SRM)
    │   └── efficientnet.py     # EfficientNet-B0 preentrenado, referencia
    ├── entrenar.py             # Entrenamiento con AMP, early stopping y checkpoints reanudables
    ├── evaluar.py              # Métricas por frame, por video y por técnica
    ├── gradcam.py              # Mapas Grad-CAM: ejemplos y promedio por técnica
    ├── metricas.py
    └── utils.py
```

## Cómo correrlo en Kaggle

El dataset ([`xdxd003/ff-c23`](https://www.kaggle.com/datasets/xdxd003/ff-c23)) ya está en Kaggle, así que no hay que descargarlo, y las corridas siguen aunque se cierre el navegador.

**Requisito:** cuenta de Kaggle verificada con el celular (*Settings → Phone verification*), para que los notebooks tengan internet y puedan clonar este repositorio.

### Notebook 1: extraer rostros (CPU)

1. En Kaggle: *Create → New Notebook → File → Import Notebook* y subir [`notebooks/kaggle_01_extraer_rostros.ipynb`](notebooks/kaggle_01_extraer_rostros.ipynb).
2. Panel derecho: **Accelerator: None**, **Internet: On**, *Add Input* → `ff-c23`.
3. *Save Version* → **Save & Run All (Commit)**.
4. Al terminar, en *Output* quedan `faces_ffpp.zip` (~5–6 GB) y `splits/`.

### Notebook 2: entrenar y evaluar (GPU)

1. Importar [`notebooks/kaggle_02_entrenar.ipynb`](notebooks/kaggle_02_entrenar.ipynb) igual que el anterior.
2. Panel derecho: **Accelerator: GPU T4 x2** (o P100), **Internet: On**, *Add Input* → *Your Work* → la salida del notebook 1.
3. *Save Version* → **Save & Run All (Commit)**.
4. Entrena la cola de experimentos con un presupuesto de 11 h por sesión (Kaggle corta a las 12 h). Si no alcanza, pausa el experimento en curso guardando el checkpoint.
5. Para continuar: nueva *Save Version*, agregando como input **la salida de la versión anterior de este mismo notebook**. Salta lo terminado y continúa lo pausado.

La GPU gratuita de Kaggle tiene una cuota de ~30 h por semana. La cola está ordenada para que lo indispensable (MesoNet, red propia y EfficientNet) termine primero.

## Uso local

### 1. Instalación

```bash
pip install -r requirements.txt
```

`src/utils.py` elige el dispositivo automáticamente: CUDA (Kaggle, GTX 1660) → MPS (Mac) → CPU.

### 2. Datos

Para correr localmente, FaceForensics++ (C23) se descarga de Kaggle ([`xdxd003/ff-c23`](https://www.kaggle.com/datasets/xdxd003/ff-c23)) con `kagglehub`. Se necesita un token de Kaggle (*Settings → API → Generate New Token*), guardado en `~/.kaggle/access_token` o en la variable `KAGGLE_API_TOKEN`. Más detalles en [`data/README.md`](data/README.md).

```python
import kagglehub
ruta = kagglehub.dataset_download("xdxd003/ff-c23")
```

### 3. Extraer rostros

```bash
python -m src.data.extraer_rostros --dataset ffpp --raiz <ruta de kagglehub> --salida data/faces/ffpp --procesos 8
```

- Detecta sola la estructura de carpetas (la de Kaggle o la del script oficial) e ignora `DeepFakeDetection`.
- Toma 32 frames de cada video real y 8 de cada falso: como hay 4 técnicas, las clases quedan balanceadas (~32 mil rostros de cada una).
- Guarda en PNG para no agregar compresión que confunda a la rama de frecuencia.
- Si se interrumpe, al volver a correr el comando continúa donde iba.
- Si el dataset trae **FaceShifter**, la extrae pero **nunca se usa para entrenar**: queda como técnica no vista (E3b).

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

En Kaggle, [`notebooks/kaggle_02_entrenar.ipynb`](notebooks/kaggle_02_entrenar.ipynb) corre todos los experimentos y arma la tabla de resultados.

### 6. Grad-CAM y demo

```bash
python -m src.gradcam --checkpoint runs/dos_ramas/mejor.pt --datos data/faces/ffpp \
    --csv data/splits/ffpp/test.csv --salida results/gradcam/dos_ramas
python -m app.demo --checkpoint runs/dos_ramas/mejor.pt runs/meso4/mejor.pt
```

## Experimentos

| ID | Qué mide | Cómo |
|---|---|---|
| E1 | Rendimiento dentro de FF++ | `evaluar` en `test.csv` |
| E2 | Aporte de cada componente | Variantes `solo_espacial`, `solo_frecuencia`, `dos_ramas_sin_se`, `dos_ramas_srm` |
| E3 | Técnica no vista | `entrenar --metodos` con 3 de las 4 técnicas |
| E3b | Técnica nunca vista (FaceShifter) | `evaluar --metodos FaceShifter` |
| E4 | Otro dataset (trabajo futuro, no se corrió) | `evaluar` en Celeb-DF v2 |
| E5 | Interpretabilidad | `gradcam` |

El Grad-CAM se validó con una prueba controlada: se entrenó una red donde lo "falso" era una mancha puesta a propósito en la boca, y el mapa la localiza.

![Validación del Grad-CAM](docs/figuras/validacion_gradcam.png)

## Modelos

| `--modelo` | Descripción | Parámetros |
|---|---|---|
| `meso4` | MesoNet, línea base desde cero | 24 mil |
| `dos_ramas` | **Red propia**: espacial + frecuencia (FFT), con atención SE | 1,63 M |
| `solo_espacial` | Ablación: sin rama de frecuencia | 1,18 M |
| `solo_frecuencia` | Ablación: sin rama espacial | 0,45 M |
| `dos_ramas_sin_se` | Ablación: sin Squeeze-and-Excitation | 1,48 M |
| `dos_ramas_srm` | Variante: rama de frecuencia con filtros SRM en lugar de FFT | 1,63 M |
| `efficientnet_b0` | Referencia preentrenada en ImageNet (usar `--lr 3e-4`) | 4,0 M |

La red propia recibe la misma imagen RGB que los demás modelos y calcula por dentro la entrada de la rama de frecuencia:

![Entradas de la rama de frecuencia](docs/figuras/rama_frecuencia_ejemplo.png)

En la fila de abajo, el *upsampling* (lo que hacen los generadores de deepfakes) deja "réplicas" en el espectro de Fourier, que no se notan en la imagen.

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics) (versión C23 en [Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23)): entrenamiento, validación y prueba.
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics): prueba de generalización prevista (trabajo futuro).

## Créditos de terceros

- División oficial de FaceForensics++ (`data/splits/ffpp_oficial/`), tomada del [repositorio de FaceForensics](https://github.com/ondyari/FaceForensics).
- Detector de rostros YuNet (`src/data/pesos/`), del [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (licencia MIT).

## Integrantes

- Julián Lara Aristizábal
- Luis Alfonso Agudelo Ramirez
