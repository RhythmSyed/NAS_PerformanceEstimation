import numpy as np 
from latency_encoding import *
import pandas as pd 
import csv

import torch.nn as nn
import torch
import torch.nn.functional as F
import math 

#Section 1: Read csv data and transform into a pandas dataset 
df = pd.read_csv("latency_dataset.csv", usecols = ["child_arch", "latency"])


#data = df.transform([lambda m : latency_encoding(eval(m)), lambda m : torch.tensor(float(m))])
data = df.transform([lambda m : latency_encoding(eval(m)), lambda m : torch.tensor(float(m))])
#Not sure why <lambda is printed out when I try it with the transform function>


"""
data = pd.DataFrame(columns=["child_arch", "latency"])

for index, row in df.iterrows():
	
	value = row['child_arch']
	
	if value == "child_arch":
		print("Passed")
		pass
	else:
		value = eval(value)
		data = data.append({'child_arch': latency_encoding(value), 'latency': torch.tensor(float(row['latency']))}, ignore_index=True)

"""


#Section 2: Create pytorch datasets
no_rows = len(data.index)

training_size = math.floor(0.7*no_rows)
validation_size = math.floor(0.15*no_rows)
test_size = no_rows - training_size - validation_size

training_data = data.iloc[:training_size,:] 
validation_data = data.iloc[training_size:training_size+validation_size,:]
test_data = data.iloc[training_size+validation_size:,:]


from torch.utils.data import Dataset

"""

Rythm: training_data, validation_data and test_data are pandas dataframes
I need to convert it into a pytorch friendly format so that I can run the model 

When I tried to directly using it on a dataloader

train_loader = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle=True)

I was getting an error. I realised that this was probably because training_data was not in the write format of torch.utils.data.Dataset
for the dataloader to work.

so I created my own Dataset class as OFA did, but I still got errors, I made some changes in getitem so that indexing 
of pandas dataset is possible, by using iloc

but I still got errors when trying to enumerate through the dataloader, so I am not sure what I should do. 
"""

class Dataset(Dataset):
    def __init__(self, population, latency):
        self.population = population
        self.latency = latency

    def __len__(self):
        return len(self.population)

    def __getitem__(self, index):
        x = self.population.iloc[index]
        y = self.latency.iloc[index]
        return x, y

print(torch.tensor(training_data["child_arch"].values))


#print(torch.tensor(training_data["child_arch"].values))

train_dataset = Dataset(training_data["child_arch"], training_data["latency"]) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)

#for i, data in enumerate(train_loader, 0):
for inputs,target in train_loader:
	print("Hello")


"""
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle=True)
	
"""


"""
#Section 3: Define the basic Deep Learning model

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


net = LatencyPredictor()


#Section 4: Define the Loss function and optimizer - 10 minutes
import torch.optim as optim

class RMSELoss(nn.Module):
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


criterion = RMSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Section 5: Train the network - 20 minutes

n_epochs = 100

for epoch in range(n_epochs):
        train_loss, val_loss = [], []
        # train
        for inputs, targets in train_loader:
            #inputs = model.get_feats(inputs)
            #inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # validate
        for inputs, targets in validation_loader:
            #inputs = model.get_feats(inputs)
            #inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
        

        print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss),
               "Valid Loss: ", np.mean(val_loss))
"""

#Section 6: Test the network and find the accuracy - 30 minutes


#Debugging - 2 hours :(