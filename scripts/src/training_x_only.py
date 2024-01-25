#!/usr/bin/env python
# coding: utf-8


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# /!\ HOW TO LAUNCH THE SCRIPT /!\
#      In a terminal in the directory of the script write this in the command prompt : python3 training.py dataset_name category_name batch_size n_epochs
#
#      dataset_name: string, folder where dataset is stored
#      category_name : string, name of the prediction feature
#      batch_size: int, size of the batches
#      n_epochs: int, number of epcohs to train the model
#     
#      Note : Check that you have installed all the required packages
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from utils.xy_dataset import XYDataset
import sys
import warnings
from sklearn.model_selection import ShuffleSplit
import cv2
from utils.utils import *
import numpy as np
import pandas as pd
import random


def main(TASK, CATEGORIES, BATCH_SIZE, N_EPOCHS):
    # Path to the annotated dataset
    CATEGORIES = [CATEGORIES] # Adapt to the right format, list exigée
    print(TASK)
    TRANSFORMS = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #dataset = XYDataset(TASK , CATEGORIES, TRANSFORMS, random_hflip=True)
    dataset = XYDataset(TASK , CATEGORIES, TRANSFORMS, random_hflip=False)
    random.shuffle(dataset.annotations)
    dataset.annotations = dataset.annotations


    # Setup the model
    device = torch.device('cpu')
    #output_dim = 2 * len(dataset.categories)  # x, y coordinate for each category
    output_dim = len(dataset.categories)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, output_dim)
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    train_eval(True, BATCH_SIZE, N_EPOCHS, model, dataset, optimizer, device)
    #Save the model
    torch.save(model.state_dict(), 'road_following_model.pth')

    # Optimize the model
    """
    from torch2trt import torch2trt

    data = torch.zeros((1, 3, 224, 224)).cuda().half()
    model_trt = torch2trt(model, [data], fp16_mode=True)
    torch.save(model_trt.state_dict(), 'road_follower_trt.pth')
    """


def train_eval(is_training, BATCH_SIZE, N_EPOCHS, model, dataset, optimizer, device):

    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        if is_training:
            model = model.train()
        else:
            model = model.eval()

        while N_EPOCHS > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, category_idx, xy in tqdm(iter(train_loader)):
                # send data to device
                images = images.to(device)
                xy = xy.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute MSE loss over x, y coordinates for associated categories
                loss = 0.0
                for batch_idx, cat_idx in enumerate(list(category_idx.flatten())):
                    loss += torch.mean((outputs[batch_idx][2 * cat_idx:2 * cat_idx+2] - xy[batch_idx])**2)
                loss /= len(category_idx)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                count = len(category_idx.flatten())
                i += count
                sum_loss += float(loss)
                # progress_widget.value = i / len(dataset)
                # loss_widget.value = sum_loss / i
                print(f"Loss: {sum_loss / i}")
                
            if is_training:
                N_EPOCHS = N_EPOCHS - 1
            else:
                break
    except Exception as e:
        print(e)
        pass
    model = model.eval()



# Script input: task, category, batch_size, n_epochs
warnings.filterwarnings("ignore", category=UserWarning)
task, category, batch_size, n_epochs = sys.argv[1:5]
batch_size = int(batch_size)
n_epochs = int(n_epochs)


if __name__ == "__main__":
    main(task, category, batch_size, n_epochs)
