import torchvision
import gzip
import os
import numpy as np
import binascii
import struct


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

trainData = torchvision.datasets.MNIST('./datasets/', train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST('./datasets/', train=False, transform=transform)


folder_path = './datasets/MNIST/raw/'
data_name = 'train-images-idx3-ubyte'

with open(os.path.join(folder_path, data_name),'rb') as f:
    info_np = np.fromfile(file=f, dtype='>i4', count=4)

data_name = 'train-images-idx3-ubyte.gz'
with gzip.open(os.path.join(folder_path, data_name),'rb') as data_path:
        x_train = np.frombuffer(
            data_path.read(), np.uint8, offset=0
        )
print(1)

def _load_data(folder_path, data_name, label_name):

    with gzip.open(os.path.join(folder_path, data_name), 'rb') as data_path:
        x_train = np.frombuffer(
            data_path.read(), np.uint8, offset=16
        )

    with gzip.open(os.path.join(folder_path, label_name), 'rb') as label_path:
        y_train = np.frombuffer(
            label_path.read(), np.uint8, offset=16
        )

