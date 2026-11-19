# Detección de deepfakes con redes neuronales

Proyecto del curso **Redes Neuronales y Aprendizaje Profundo** (Universidad EAFIT).

Diseñamos y entrenamos desde cero una red neuronal convolucional de dos ramas, **espacial** y **de frecuencia**, que clasifica rostros extraídos de videos como reales o manipulados. Luego la comparamos con MesoNet y con EfficientNet-B0 preentrenado, y estudiamos qué tan bien generaliza a técnicas de manipulación y datasets que no vio durante el entrenamiento.

📄 **[Anteproyecto](docs/anteproyecto.md)**: problema, objetivos, hipótesis, metodología, cronograma y riesgos.

## Estado

🟡 **Fase de planeación.** El código se agregará a partir de la semana 1.

## Estructura del repositorio

```
├── data/        # Datos (no versionados): instrucciones de descarga y listas de división
├── docs/        # Anteproyecto, informe y figuras
├── notebooks/   # Exploración, entrenamiento en Colab y análisis
├── results/     # Métricas y gráficas de los experimentos
└── src/         # Código: preprocesamiento, modelos, entrenamiento y evaluación
```

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics): entrenamiento, validación y prueba.
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics): prueba de generalización.

Ambos requieren solicitar acceso y aceptar sus términos de uso. Los datos **no** se suben a este repositorio; ver [`data/README.md`](data/README.md).

## Integrantes

- Julián Lara Aristizábal
- _[agregar integrantes]_
