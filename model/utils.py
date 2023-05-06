import os
import torch
from torchvision import datasets, transforms

def get_images(data_dir='./model/dataset'):
    # Define the transform to apply to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Define the dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Define the mapping from label names to numerical labels
    name_to_label = {name: label for label, name in enumerate(os.listdir(data_dir))}

    return dataset, name_to_label
