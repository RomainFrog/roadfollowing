import torch
import torchvision
import cv2
import os
from utils.utils import *
import uuid
import sys
import warnings

def predict_apex(image):
    dim = (244, 244)
    img = cv2.imread(image)

    # Redimensionner l'image
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    #Apply model
    preprocessed = preprocess(img)
    output = model(preprocessed).detach().cpu().numpy().flatten()
    
    #recalculate coordinates
    x = int(dim[0]* (output[0] / 2.0 + 0.5))
    y = int(244/2) #int(dim[1]* (output[1] / 2.0 + 0.5))
    
    filename = "%d_%d_%s.jpg" % (x,y,str(uuid.uuid1()))   
    circled_image = cv2.circle(img, (x, y), 8, (0, 255, 0), 3) 
    cv2.imwrite(output_folder + "/" + filename, circled_image)


warnings.filterwarnings("ignore", category=UserWarning)
input_folder, output_folder, model_file = sys.argv[1:4]
output_dim = 1 #2
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, output_dim)
model.load_state_dict(torch.load(model_file))


for filename in os.listdir(input_folder):

    os.makedirs(output_folder, exist_ok=True)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(input_folder + "/" + filename)
        predict_apex(input_folder + "/" + filename)
