import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import TestDataset
from torchvision import datasets, models, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = './data'
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
image_datasets = {
    x: datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
class_names = image_datasets['val'].classes


def get_model(type, weights):
    if type == 'resnet50':
        model = models.resnet50()
    elif type == 'resnet101':
        model = models.resnet101()
    elif type == 'resnext101_64x4d':
        model = models.resnext101_64x4d()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)
    model = model.to(device)
    model.load_state_dict(
        torch.load(weights, map_location=device, weights_only=True)
    )
    model = model.cpu()
    return model


if __name__ == '__main__':
    test_dir = os.path.join(root_dir, 'test')
    test_dataset = TestDataset(test_dir, transform=data_transforms['test'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=32, shuffle=False, num_workers=4
    )

    weights_list = [
        ('resnet101', './result/0.pth'),
        ('resnet101', './result/1.pth'),
        ('resnet101', './result/2.pth'),
        ('resnet101', './result/3.pth'),
        ('resnet101', './result/4.pth')
    ]

    ensemble_models = []
    for type, weights in weights_list:
        ensemble_models.append(get_model(type, weights))

    results = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for inputs, names in pbar:
            inputs = inputs.to(device)
            votes = []
            for model in ensemble_models:
                model_gpu = model.to(device)
                model_gpu.eval()
                outputs = model_gpu(inputs)
                _, preds = torch.max(outputs, 1)
                votes.append(preds.cpu().numpy())
                model_gpu.cpu()
            # Transpose votes to get predictions per sample, then majority vote
            votes = np.array(votes)  # Shape: (num_models, batch_size)
            final_preds = []
            for col in votes.T:
                counts = np.bincount(col, minlength=len(class_names))
                final_preds.append(np.argmax(counts))
            results.extend(zip(names, [class_names[p] for p in final_preds]))

    df = pd.DataFrame(results, columns=['image_name', 'pred_label'])
    df.to_csv('./tmp.csv', index=False)
    print('end of main')
