import numpy as np
from latency_encoding import *
import pandas as pd
import csv
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import torch.optim as optim

model_path = 'checkpoints/latency_predictor/checkpoint.pt'


class LatencyDataset(Dataset):
    def __init__(self, population, latency):
        self.population = population
        self.latency = latency

    def __len__(self):
        return len(self.population)

    def __getitem__(self, index):
        x = self.population.iloc[index]
        y = self.latency.iloc[index]
        return x, y


class LatencyPredictor(nn.Module):
    def __init__(self):
        super(LatencyPredictor, self).__init__()
        self.fc1 = nn.Linear(128, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RMSELoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def data_preprocessing():
    # Section 1: Read csv data and transform into a pandas dataset
    df = pd.read_csv("latency_dataset.csv", usecols=["child_arch", "latency"])
    data = df.transform([lambda m: latency_encoding(eval(m)), lambda m: torch.tensor(float(m))])
    data_latency = data['latency']['<lambda>']
    data_child_arch = data['child_arch']['<lambda>']
    data = pd.DataFrame({'child_arch': data_child_arch, 'latency': data_latency})

    # Section 2: Create pytorch datasets
    no_rows = len(data.index)
    training_size = math.floor(0.7 * no_rows)
    validation_size = math.floor(0.15 * no_rows)
    test_size = no_rows - training_size - validation_size

    training_data = data.iloc[:training_size, :]
    validation_data = data.iloc[training_size:training_size + validation_size, :]
    test_data = data.iloc[training_size + validation_size:, :]

    return training_data, validation_data, test_data


def main():
    # Section 1: Read csv data and transform into a pandas dataset
    # Section 2: Create pytorch datasets
    training_data, validation_data, test_data = data_preprocessing()
    train_dataset = LatencyDataset(training_data["child_arch"], training_data["latency"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataset = LatencyDataset(validation_data["child_arch"], validation_data["latency"])
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

    # Section 3: Define the basic Deep Learning model
    model = LatencyPredictor()

    # Section 4: Define the Loss function and optimizer
    criterion = RMSELoss()
    optim_config = {
        'lr': 1e-3,
        'weight_decay': 1e-2,
    }
    optimizer = optim.SGD(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    # Section 5: Train the network
    n_epochs = 100
    for epoch in range(n_epochs):
        train_loss, val_loss = [], []
        # train
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # validate
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())

        print('Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}'.format(epoch, train_loss[-1], val_loss[-1]))

    # Save Model
    torch.save(model.state_dict(), model_path)


def test_model():
    # Section 6: Test the network and find the latency
    checkpoint = torch.load(model_path)
    model = LatencyPredictor()
    model.load_state_dict(checkpoint)
    model.eval()
    sample_child_arch = {'wid': None, 'ks': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
                         'e': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'd': [3, 2, 2, 3, 3], 'r': [176]}
    sample_input = latency_encoding(sample_child_arch)
    predicted_latency = model(sample_input)
    print('Child Arch: {}, Predicted Latency: {}'.format(sample_child_arch, predicted_latency))


if __name__ == '__main__':
    main()
    # test_model()
