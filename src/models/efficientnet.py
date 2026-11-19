"""EfficientNet-B0 preentrenado en ImageNet — referencia de rendimiento ("techo").

No es parte de la propuesta: sirve para medir qué tan cerca queda la red propia,
entrenada desde cero, de un modelo grande preentrenado (pregunta P1).

Se reemplaza la última capa por una salida de un logit y se ajusta toda la red
(fine-tuning). Se recomienda un learning rate más bajo que para los modelos desde
cero, por ejemplo --lr 3e-4.
"""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0(nn.Module):
    def __init__(self, tam_entrada: int = 224, preentrenado: bool = True):
        super().__init__()
        pesos = EfficientNet_B0_Weights.IMAGENET1K_V1 if preentrenado else None
        self.red = efficientnet_b0(weights=pesos)
        entrada = self.red.classifier[1].in_features
        self.red.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(entrada, 1))

    def forward(self, x):
        return self.red(x).squeeze(1)

    def capa_gradcam(self) -> nn.Module:
        return self.red.features[-1]
