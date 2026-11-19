# Datos

Los videos y los rostros extraídos **no se versionan** en este repositorio por su tamaño y por los términos de uso de los datasets. Aquí solo se guardan instrucciones y las listas de división (`splits/`).

## Cómo obtener los datos

1. **FaceForensics++**: llenar el formulario indicado en el [repositorio oficial](https://github.com/ondyari/FaceForensics). Al ser aprobado se recibe un script de descarga. Descargar solo la versión comprimida **c23** (y opcionalmente c40 para las pruebas de robustez).
2. **Celeb-DF v2**: solicitar acceso desde el [repositorio oficial](https://github.com/yuezunli/celeb-deepfakeforensics).

## Estructura local esperada

```
data/
├── raw/          # Videos originales descargados
├── faces/        # Recortes de rostros (224×224) generados por el preprocesamiento
└── splits/       # Listas de videos por partición (train / val / test) — sí se versionan
```
