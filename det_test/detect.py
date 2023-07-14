import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
import os
from PIL import Image

import time

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


MIN_SIZE = 800
image_name = "demo.jpg"
output_path = "outputs"
if not os.path.exists(output_path):
    os.mkdir(output_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,min_size=MIN_SIZE)

image = Image.open(image_name)
model.eval().to(device)

start = time.time()
output = detect_utils.predict(image, model, device)
end = time.time()

print("calculation time: %f", end-start)

image = detect_utils.draw_boxes(image=image, output=output, predict_threshold=0.7)
save_name = f"{image_name.split('.')[0]}_{image_name}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
