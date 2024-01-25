#!/usr/bin/env python
# coding: utf-8

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# /!\ HOW TO LAUNCH THE SCRIPT /!\
#      In a terminal in the directory of the script write this in the command prompt : python3 testing_prediction_accuracy.py model_location dataset_name category_name
#       
#       model_location : string, folder where the model is saved
#      dataset_name: string, folder where dataset is stored
#      category_name : string, name of the prediction feature
#     
#      Note : Check that you have installed all the required packages
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from utils.utils import *
import torch
import torchvision
from utils.xy_dataset import XYDataset
import pandas as pd
import sys
import math
from get_apex_from_yolo import * 
from ultralytics import YOLO



#Loading the parameters needed
model_file, TASK, CATEGORIES = sys.argv[1:4]

model = YOLO(model_file)

#Loading the dataset
CATEGORIES = [CATEGORIES] # Adapt to the right format, list exigée
print(TASK)
TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#dataset = XYDataset(TASK , CATEGORIES, TRANSFORMS, random_hflip=True)
test_dataset = XYDataset(TASK , CATEGORIES, TRANSFORMS, random_hflip=True)

print(len(test_dataset.annotations), " elements in the test dataset")


#Processing the distances between predicted apex and the real one
print("Pixel errors descriptions on (X,Y)")
distances = []
distances_y = []
for item in test_dataset.annotations:
    a = np.array((item['x'], item['y']))
    b = get_apex(model, item['image_path'])
    distances.append(np.linalg.norm(a-b))
    distances_y.append(np.linalg.norm(a[1]-b[1]))

#Some insights about distances
desc = pd.Series(distances).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])


#Processing the distances between predicted apex and the real one
print('Pixel errors description on Y:')


#Some insights about distances
desc = pd.Series(distances_y).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])



