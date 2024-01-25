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
import time


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
    
    return y


#Loading the parameters needed
model_file, TASK, CATEGORIES, N_MODEL = sys.argv[1:5]
N_MODEL = int(N_MODEL)

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

if N_MODEL == 0:
    model = torchvision.models.efficientnet_b0(pretrained=True)
if N_MODEL == 1:
    model = torchvision.models.efficientnet_b1(pretrained=True)
if N_MODEL == 2:
    model = torchvision.models.efficientnet_b2(pretrained=True)
if N_MODEL == 3:
    model = torchvision.models.efficientnet_b3(pretrained=True)
if N_MODEL == 4:
    model = torchvision.models.efficientnet_b4(pretrained=True)

if N_MODEL == 5:
    model = torchvision.models.efficientnet_b5(pretrained=True)
if N_MODEL == 6:
    model = torchvision.models.efficientnet_b6(pretrained=True)
if N_MODEL == 7:
    model = torchvision.models.efficientnet_b7(pretrained=True)

if N_MODEL == 11:
    model = torchvision.models.efficientnet_v2_s(pretrained=True)
if N_MODEL == 12:
    model = torchvision.models.efficientnet_v2_m(pretrained=True)
if N_MODEL == 13:
    model = torchvision.models.efficientnet_v2_l(pretrained=True)

if N_MODEL == 18:
    model = torchvision.models.resnet18(pretrained=True)
if N_MODEL == 34:
    model = torchvision.models.resnet34(pretrained=True)
if N_MODEL == 50:
    model = torchvision.models.resnet50(pretrained=True)

if N_MODEL == 100:
    model = torchvision.models.squeezenet1_1(pretrained=True)
    model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=1)
    model.num_classes = len(test_dataset.categories)   

if N_MODEL == 110:
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(4096, 2)

model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(model_file))

# model.eval()


#Processing the distances between predicted apex and the real one
distances_xy = []
distances = []

start_time = time.time()
for item in test_dataset.annotations:
    prediction = predict_apex(item['image_path'], model)
    a = np.array((item['x'], item['y']))
    b = np.array(prediction)
    distances_xy.append(np.linalg.norm(a-b))

    a = np.array((item['y']))
    b = np.array(prediction[0])
    distances.append(np.linalg.norm(a- b))
end_time = time.time()

#Some insights about distances
desc = pd.Series(distances_xy).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])

#Processing the distances between predicted apex and the real one
print('ON one dimension ONLY NOW : ')
#Some insights about distances
desc = pd.Series(distances).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])

print("Temps d'exécution : ", end_time - start_time, " secondes")