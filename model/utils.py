import os
import torch
from PIL import Image
from torchvision import datasets, transforms

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

def get_labels(data_dir='./model/dataset'):
    return {label: name for label, name in enumerate(os.listdir(data_dir))}

def get_images(data_dir='./model/dataset'):

    # Define the dataset
    dataset = datasets.ImageFolder(data_dir, transform=get_transform())

    return dataset

def get_test_images(data_dir='./model/test_dataset'):
    transformer = get_transform()
    dataset = []
    for image in os.listdir(data_dir):
        image = Image.open(f"{data_dir}/{image}")
        image = transformer(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        dataset.append(image)

    return dataset
