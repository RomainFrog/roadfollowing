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


#Function to predict the apex in a given image with the model specified
def predict_apex(image, model):
    dim = (256, 256)
    img = cv2.imread(image)

    # Redimensionner l'image
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    #Apply model
    preprocessed = preprocess(img)
    output = model(preprocessed).detach().cpu().numpy().flatten()
    
    #recalculate coordinates
    x = int(dim[0]* (output[0] / 2.0 + 0.5))
    y = int(dim[1]* (output[1] / 2.0 + 0.5))
    
    return x,y

def predict_apex_only_x(image,model):
    dim = (256, 256)
    img = cv2.imread(image)

    # Redimensionner l'image
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    #Apply model
    preprocessed = preprocess(img)
    output = model(preprocessed).detach().cpu().numpy().flatten()
    
    #recalculate coordinates
    x = int(dim[0]* (output[0] / 2.0 + 0.5))
    #y = int(dim[1]* (output[1] / 2.0 + 0.5))
    
    return x#,y

#Loading the parameters needed
model_file, TASK, CATEGORIES, N_MODEL = sys.argv[1:5]
N_MODEL = int(N_MODEL)

if N_MODEL == 0:
        model = torchvision.models.efficientnet_b0(pretrained=True)
if N_MODEL == 3:
    model = torchvision.models.efficientnet_b3(pretrained=True)
if N_MODEL == 5:
    model = torchvision.models.efficientnet_b5(pretrained=True)

if N_MODEL == 11:
    model = torchvision.models.efficientnet_v2_s(pretrained=True)
if N_MODEL == 13:
    model = torchvision.models.efficientnet_v2_l(pretrained=True)

if N_MODEL == 18:
    model = torchvision.models.resnet18(pretrained=True)
if N_MODEL == 34:
    model = torchvision.models.resnet34(pretrained=True)
if N_MODEL == 50:
    model = torchvision.models.resnet50(pretrained=True)

model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(model_file))

# model.eval()


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
distances = []
for item in test_dataset.annotations:
    a = np.array((item['y'], item['x']))
    b = np.array(predict_apex(item['image_path'], model))
    distances.append(np.linalg.norm(a-b))

#Some insights about distances
desc = pd.Series(distances).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])


#Processing the distances between predicted apex and the real one
print('ON one dimension ONLY NOW : ')
distances = []
for item in test_dataset.annotations:
    a = np.array((item['y']))
    b = np.array(predict_apex_only_x(item['image_path'], model))
    distances.append(np.linalg.norm(a- b))

#Some insights about distances
desc = pd.Series(distances).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])