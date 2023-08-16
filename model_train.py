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
import numpy as np
import torch
from torch import optim
import albumentations as A
import json
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Custom classes
sys.path.append('./modules')
from datatools import *
from losstools import loss_function_wrapper
from models import Hourglass


# In[2]:


"""
Declaring tunable hyperparameters
"""

# dataset generation hyperparameters
patch_size = 200
sigma = 9

# training hyperparameters
epochs = 2.5e3
lr = 5e-4
batch_size = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Hourglass(depth=1).to(device)
loss_func = loss_function_wrapper('gcel')
transform = A.Compose([
    A.RandomRotate90(p=1),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])
opt = torch.optim.Adam(model.parameters(), lr=lr)

# performance logger parameters
write = False
save_model = False
if write or save_model:
    folder = './active/2023.08.15 model_depth'
    model_name = 'depth={}, gce'.format(1)
if write:
    writer = SummaryWriter('./{}/runs/{}'.format(folder, model_name))


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
                          patch_size=patch_size)
exclude_list = ['20190822_movie_01_SampleOldA1_120kV_81x2048x2048_30sec_Aligned 11.48.51 AM patch_080.jpg']
train_ds, val_ds = load_temp_dataset(parent_dir, exclude_list)

# Dataloader creation
train_ds = MapDataset(train_ds)
val_ds = MapDataset(val_ds)
train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size)


# In[4]:


for epoch in range(int(epochs)):
    """Training"""
    model.train()
    total_train_loss = 0
    for x, y in train_loader:
        # transform data and send to device
        x, y = transform_data(x, y, transform)
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
            x = x.to(device)
            y = y.to(device)
            # make prediction and calculate loss
            pred = model(x)
            loss = loss_func(pred, y)
            total_val_loss += loss
            
    # calculate precision and recall
    average_precision = 0
    average_recall = 0
    
    for image_name in exclude_list:
        image_path = os.path.join(parent_dir, 'images', image_name)
        mask_path = os.path.join(parent_dir, 'masks', image_name)
        
        val_image = np.array(Image.open(image_path).convert('L')).astype(np.float32)
        val_image /= val_image.max()
        val_mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
        val_mask /= val_mask.max()
        pred_mask = np.zeros(val_mask.shape)
        
        t = get_pr_stats(model, val_image, val_mask, patch_size, device)
        average_precision += t[0]
        average_recall += t[1]
        print("t: {}".format(t))
        
    average_precision /= len(exclude_list)
    average_recall /= len(exclude_list)

    # calculate average loss from aggregate
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    # record performance statistics and save model, if necessary
    if write:
        writer.add_scalar('train_loss', avg_train_loss, epoch)
        writer.add_scalar('val_loss', avg_val_loss, epoch)
        writer.add_scalar('average_precision', average_precision, epoch)
        writer.add_scalar('average_recall', average_recall, epoch)
    if save_model:
        if (epoch + 1) % 50 == 0:
            model_param_path = './{}/model_saves/{}, epoch={:05}.pth'.format(folder, model_name, epoch + 1)
            torch.save(model.state_dict(), model_param_path)

# close performance logger, if necessary
if write:
    writer.flush()
    writer.close()

