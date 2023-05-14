import torch
import torch.nn as nn
import torchvision.models as models
from .utils import *

def get_predictions(data_dir="./model/dataset",
                    test_dir="./model/test_dataset",
                    weight_dir="./model/model_weights.pth"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels_to_name = get_labels(data_dir)

    dataset, _ = get_test_images(test_dir)
    model = models.resnet50()
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 150) #No. of classes = 150
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir))
    model.eval()

    predicts=[]

    for i in range(len(dataset[:10])):
        data = dataset[i].cuda()
        output = model(data)
        conf_10, predicted_10 = torch.topk(output.data, k=10, dim=1)

        conf_10, predicted_10 = conf_10.tolist(), predicted_10.tolist()
        labels = [labels_to_name[key] for key in predicted_10[0]]
        confs = [conf for conf in conf_10[0]]
        predicts.append({
            "Test": f"{i+1}",
            "Top 10": labels,
            "Confidence Levels": confs
        })
    return predicts


