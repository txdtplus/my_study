from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch
from PIL import Image

import cv2

weights = models.ResNet50_Weights.DEFAULT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(img):

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    img_tensor = transform(img.copy())
    img_tensor = img_tensor.to(device).unsqueeze(0)

    return img_tensor


if __name__ == '__main__':

    img_path = "../datasets/mini-imagenet/images/n0211371200000622.jpg"
    imagenet_classes_path = "../datasets/imagenet_classes.txt"

    # Step 1: read images and preprocessing
    img = np.array(Image.open(img_path))
    img_tensor = preprocess_image(img)
    
    # Step 2: Initialize model with the best available weights and predict
    model = models.resnet50(weights=weights)
    model.to(device=device)
    model.eval()

    with torch.no_grad():
        predict = model(img_tensor)
    
    probabilities = torch.nn.functional.softmax(predict[0], dim=0)

    with open(imagenet_classes_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    # Step 3: Calculate the heatmap
    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=img_tensor)
    grayscale_cam = grayscale_cam[0, :, :]
    pad_width = np.int64((256-224)/2)

    # 两种方法，第一种直接resize。第二种先padding再resize. 
    # 由于图像输入之前首先resize成256大小的，然后再中心裁剪，因此先padding应该更科学一点

    # 第一种
    grayscale_cam1 = cv2.resize(grayscale_cam, dsize=img.shape[1::-1])

    # 第二种
    grayscale_cam = np.pad(grayscale_cam, pad_width=pad_width, mode='constant', constant_values=0)
    grayscale_cam2 = cv2.resize(grayscale_cam, dsize=img.shape[1::-1])
    
    # Step 4: Visualization
    img = np.float32(img) / 255

    cam_image1 = show_cam_on_image(img, grayscale_cam1, use_rgb=True)
    im1 = Image.fromarray(cam_image1)
    im1.show()

    cam_image2 = show_cam_on_image(img, grayscale_cam2, use_rgb=True)
    im2 = Image.fromarray(cam_image2)
    im2.show()
