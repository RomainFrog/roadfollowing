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
from skimage import io
from skimage.transform import resize
from PIL import Image


def predict_apex(image,model):
    """
    Function to predict the apex in a given image with the model specified
    """
    dim = (244, 244)
    image = Image.open(image)
    image = TRANSFORMS(image)
    image = image.unsqueeze(0)
    
    #Apply model
    output = model(image)
    #recalculate coordinates
    predicted_x = int(dim[0] * (output / 2.0 + 0.5))
    #y = int(dim[1]* (output[1] / 2.0 + 0.5))
    
    return predicted_x

#Loading the parameters needed
model_file, TASK, CATEGORIES = sys.argv[1:4]

#Loading the model
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 1)
model.load_state_dict(torch.load(model_file))


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
test_dataset = XYDataset(TASK , CATEGORIES, TRANSFORMS, random_hflip = False)

print(len(test_dataset.annotations), " elements in the test dataset")

#Processing the distances between predicted apex and the real one
distances = []
for item in test_dataset.annotations:
    print(item)
    #a = np.array((item['x'], item['y']))
    a = np.array((item['x']))
    b = np.array(predict_apex(item['image_path'], model))
    print(a,b)
    distances.append(np.linalg.norm(a-b))
    


#Some insights about distances
desc = pd.Series(distances).describe()
for idx in desc.index:
    print(idx, " : ", desc[idx])

