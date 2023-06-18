import torchvision.transforms as transforms
import cv2
import torchvision.models as models
import torch
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

transform = transforms.Compose([
    transforms.ToTensor(),
])


def predict(image:torch.Tensor, model:models, device):

    image = transform(image).to(device)
    image = image.unsqueeze(0)

    outputs = model(image)

    return outputs[0]


def draw_boxes(image, output, predict_threshold):

    pred_classes = [coco_names[i] for i in output['labels'].cpu().numpy()]
    pred_scores = output['scores'].detach().cpu().numpy()
    pred_bboxes = output['boxes'].detach().cpu().numpy()

    boxes = pred_bboxes[pred_scores >= predict_threshold].astype(np.int32)
    scores = pred_scores[pred_scores >= predict_threshold]

    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[output['labels'][i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )

        score = '%.2f'%scores[i]
        text = pred_classes[i] + " score: " + score
        cv2.putText(image, text, (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image