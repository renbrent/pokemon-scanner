import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels_to_name = get_labels()

dataset, _ = get_test_images()
model = models.resnet50()
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 150) #No. of classes = 150
model = model.to(device)
model.load_state_dict(torch.load('./model/model_weights.pth'))
model.eval()

predicts=[]

for data in dataset:
    data = data.cuda()
    output = model(data)
    conf_10, predicted_10 = torch.topk(output.data, k=10, dim=1)

    conf_10, predicted_10 = conf_10.tolist(), predicted_10.tolist()
    for i in range(len(conf_10[0])):
        print(f"{labels_to_name[predicted_10[0][i]]:12} confidence:{conf_10[0][i]:.2f}%")
        print("======================")


