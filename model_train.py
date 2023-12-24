#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Importing dependenceis
    - Outside packages
    - Custom classes
"""

# Outside packages
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from tensorboardX import SummaryWriter

# Custom classes
sys.path.append('./modules')
from model_tools import Hourglass
from temp_dataset_tools import make_dataset
from loss_tools import loss_function_wrapper
from data_tools import MapDataset, transform_data


# In[2]:


"""
Declaring tunable hyperparameters
"""

# dataset generation hyperparameters
label = 1
sigma = 9

# training hyperparameters
lr = 5e-4
depth = 1
epochs = 2.5e3
batch_size = 1

# create training objects
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Hourglass(depth=depth).to(device)
loss_func = loss_function_wrapper('gcel')
transform = A.Compose([
    A.RandomRotate90(p=1),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])
opt = torch.optim.Adam(model.parameters(), lr=lr)

# performance logger parameters
test_category = '2023.12.24 I shouldn\'t be working right now'
model_name = 'gcel, label={}'.format(label)

write = False
save_model = False
if write or save_model:
    if os.path.exists('./active') == False:
        os.mkdir('./active')
    os.mkdir(os.path.join('./active', test_category, 'model_saves'))
    os.mkdir(os.path.join('./active', test_category, 'runs'))
if write:
    writer = SummaryWriter(os.path.join('./active', test_category, 'runs', model_name))


# In[3]:


"""
Constructing a pipeline to control how data is used by the model
    - Temporary dataset creation: split dataset's patches into training and validation groups
        * use 20190822...11.48.51 AM patch_[range(5, 76, 5)].jpg for training
        * use 20190822...11.48.51 AM patch_[80].jpg for validation
    - Dataloader creation: package lists of patches into torch Dataloaders
"""

# Temporary dataset creation
parent_dir = make_dataset('./dataset',
                         sigma=sigma,
                         label=label)
exclude_list = ['20190822_movie_01_SampleOldA1_120kV_81x2048x2048_30sec_Aligned 11.48.51 AM patch_080.jpg.npy']
train_ds, val_ds = ([], [])
for image_name in os.listdir(os.path.join(parent_dir, 'images')):
    if image_name[-4:] != '.npy':
        continue
    image, mask = np.load(os.path.join(parent_dir, 'images', image_name)), np.load(os.path.join(parent_dir, 'masks', image_name))
    if image_name in exclude_list:
        val_ds.append([image, mask])
    else:
        train_ds.append([image, mask])

# Dataloader creation
train_ds = MapDataset(train_ds)
val_ds = MapDataset(val_ds)
train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size)


# In[4]:


for epoch in tqdm(range(int(epochs))):
    """Training"""
    model.train()
    total_train_loss = 0
    for x, y in train_loader:
        # transform data and send to device
        x, y = transform_data(x, y, transform)
        if len(x.shape) == 3:
            x, y = x[:, None, :, :], y[:, None, :, :]
        x = x.to(device)
        y = y.to(device)
        # make prediction and calculate loss
        pred = model(x)
        loss = loss_func(pred, y)
        total_train_loss += loss
        # backpropogration
        opt.zero_grad()
        loss.backward()
        opt.step()

    """Validation"""
    with torch.no_grad():
        model.eval()
        total_val_loss = 0
        for x, y in val_loader:
            # transform data and send to device
            x, y = transform_data(x, y, transform)
            if len(x.shape) == 3:
                x, y = x[:, None, :, :], y[:, None, :, :]
            x = x.to(device)
            y = y.to(device)
            # make prediction and calculate loss
            pred = model(x)
            loss = loss_func(pred, y)
            total_val_loss += loss

    # calculate average loss from aggregate
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    # record performance statistics and save model, if necessary
    if write:
        writer.add_scalar('train_loss', avg_train_loss, epoch)
        writer.add_scalar('val_loss', avg_val_loss, epoch)

    if save_model:
        if (epoch + 1) % 50 == 0:
            model_save_path = os.path.join('./active', test_category, 'model_saves', model_name+" epoch={:05}.pth".format(epoch+1))
            torch.save(model.state_dict(), model_save_path)

# close performance logger, if necessary
if write:
    writer.flush()
    writer.close()

