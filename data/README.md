# Datos

Los videos y los rostros extraídos **no se versionan** en este repositorio por su tamaño y por los términos de uso de los datasets. Aquí solo se guardan instrucciones y las listas de división (`splits/`).

## Cómo obtener los datos

1. **FaceForensics++**: llenar el formulario indicado en el [repositorio oficial](https://github.com/ondyari/FaceForensics). Al ser aprobado se recibe un script de descarga. Descargar solo la versión comprimida **c23** (y opcionalmente c40 para las pruebas de robustez).
2. **Celeb-DF v2**: solicitar acceso desde el [repositorio oficial](https://github.com/yuezunli/celeb-deepfakeforensics).

Estructura que deja el script de FaceForensics++ (es la que espera `src/data/extraer_rostros.py`):

```
FaceForensics/
├── original_sequences/youtube/c23/videos/000.mp4 …
└── manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/000_003.mp4 …
```

Comando de descarga sugerido (una vez aprobado el acceso):

```bash
python download-FaceForensics.py /ruta/FaceForensics -d original -c c23 -t videos
for m in Deepfakes Face2Face FaceSwap NeuralTextures; do
  python download-FaceForensics.py /ruta/FaceForensics -d $m -c c23 -t videos
done
```

## Estructura local esperada

```
data/
├── raw/          # Videos originales descargados
├── faces/        # Recortes de rostros (224×224) generados por el preprocesamiento
└── splits/       # Listas de videos por partición (train / val / test) — sí se versionan
```
