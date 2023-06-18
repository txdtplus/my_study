import torch
import torchvision.models as models
import numpy as np
from PIL import Image

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

img_path = "../datasets/mini-imagenet/images/n0211371200000622.jpg"

img = np.array(Image.open(img_path))