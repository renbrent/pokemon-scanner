import torch
import torch.nn as nn
import torchvision.models as models
from model.utils import *
from .labels import POKEMON_NAMES

WEIGHT_DIR = "../model/model_weights.pth"
UPLOADS = "../flask_app/static/uploads/"


def get_predictions(test_dir="./model/test_dataset"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset, _ = get_test_images(test_dir)
    model = models.resnet50()
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 150)  # No. of classes = 150
    model = model.to(device)
    model.load_state_dict(torch.load(WEIGHT_DIR))
    model.eval()

    predicts = []

    for i in range(len(dataset[:10])):
        data = dataset[i].cuda()
        output = model(data)
        conf_10, predicted_10 = torch.topk(output.data, k=10, dim=1)

        conf_10, predicted_10 = conf_10.tolist(), predicted_10.tolist()
        labels = [POKEMON_NAMES[key] for key in predicted_10[0]]
        confs = [conf for conf in conf_10[0]]
        predicts.append({"Top 10": labels, "Confidence Levels": confs})
    return predicts


def model_predict(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = process_image(f"{UPLOADS}{image}")
    model = models.resnet50()
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 150)  # No. of classes = 150
    model = model.to(device)
    model.load_state_dict(torch.load(WEIGHT_DIR))
    model.eval()

    data = image.cuda()
    output = model(data)
    conf_10, predicted_10 = torch.topk(output.data, k=10, dim=1)

    conf_10, predicted_10 = conf_10.tolist(), predicted_10.tolist()
    predict = [{"Rank": i+1, "Pokemon": POKEMON_NAMES[key], "Confidence Level": conf} for i, (key, conf) in enumerate(zip(predicted_10[0], conf_10[0]))]

    return predict
