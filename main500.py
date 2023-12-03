# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
from torch.utils.data import DataLoader
import glob
import random


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

file = 'archive/YUM.csv'



# %%

files = glob.glob("archive/*.csv")
# random.shuffle(files)
# files

# %% [markdown]
# <h2>Model</h2>

# %% [markdown]
# <h2>Training</h2>

# %%
from Modules.train import train_model

# %%

# Load data into pytorch dataset

from Modules.dataset_class import TimeSeriesDataset
from Modules.preprocess import process_data
def train_on_file(file, model, num_epochs, loss_function, optimizer, device):

    print("PROCESSING "+ file)
    _, _, X_train, X_test, y_train, y_test, _ = process_data(file)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # create batches
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = train_model(num_epochs, model, loss_function, optimizer, train_loader,test_loader, device)
    return model

# %%
def train_all(files, input_model, num_epochs, loss_function, optimizer, device):
    model = input_model
    save = 1
    if torch.cuda.is_available():
        model.cuda()
        model.gradient_checkpointing_enable()
    for file in files:
        model = train_on_file(file, model, num_epochs, loss_function, optimizer, device)
        # if save % 5 == 0:
        #     torch.save(model, 'checkpoints/forecast'+str(save)+'.pt')
        # save += 1

    torch.save(model, 'forecast_rnn.pt')

# %%

from Modules.model import ElmanRNN

model = ElmanRNN(30, 1, 64, 16, 1,)
learning_rate = 0.001
num_epochs = 10
loss_function = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

files = sorted(glob.glob("archive/*.csv"))[:10]
# files = glob.glob("../..archive/*.csv")
# random.shuffle(files)

files_list_path = "filenames.txt"
with open(files_list_path, 'w') as file:
    # Write each element of the string array to the file
    for item in files:
        file.write(item + '\n')



train_all(files, model, num_epochs, loss_function, optimizer, device)


