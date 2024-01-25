#! /usr/bin/env python3

from jetcam.csi_camera import CSICamera
# from jetcam.usb_camera import USBCamera

camera = CSICamera(width=224, height=224, capture_fps=60)
# camera = USBCamera(width=224, height=224)

camera.running = True

import torch
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('models/road_following_model_2_trt.pth'))
model_trt.eval()

binary_classifier_trt = TRTModule()
binary_classifier_trt.load_state_dict(torch.load('models/binary_classifier_2_trt.pth'))
binary_classifier_trt.eval()

import ipywidgets
import traitlets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

def flip_and_jpeg(img):
    return bgr8_to_jpeg(cv2.flip(img, -1))

# create image preview
#image_with_apex_widget = ipywidgets.Image(width=camera.width, height=camera.height)
#display(image_with_apex_widget)

import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

import numpy as np
from remote_control.control_sender import ControlSender
from remote_control.image_sender import ImageSender
from smbus import SMBus

sender = ControlSender("10.0.1.1", 1234)
image_sender = ImageSender("10.0.1.1", 1235)

distance = 0
frames_per_ping = 5
frames_since_ping = 0
print("COUCOUOUUUU")
# Setup the ultrasonic module
#bus = SMBus(1) # Open I2C bus 1
#bus.write_byte_data(0x70, 0x01, 25) # Reduce max gain to 352
#bus.write_byte_data(0x70, 0x02, round(3/0.043)) # Reduce max distance to 3m
#bus.write_byte_data(0x70, 0x00, 0x51) # Request a first ranging in centimeters

while True:
    image = cv2.flip(camera.value,-1)
    image_with_apex = image.copy()
    image = preprocess(image).half()
    raw_road_pred = binary_classifier_trt(image)
    is_road_pred = torch.sign(raw_road_pred[0][1]).item()
    
    if is_road_pred == 1:
    
        apex_pred = model_trt(image).detach().cpu().numpy().flatten()   
        x = apex_pred[0]
        y = apex_pred[1]

        x = int(camera.width * (x / 2.0 + 0.5))
        y = int(camera.height * (y / 2.0 + 0.5))
        image_with_apex = cv2.circle(image_with_apex, (x, y), 8, (255, 0, 0), 3)
        #image_with_apex_widget.value=bgr8_to_jpeg(image_with_apex)
        apex = float(apex_pred[0])
        distance = 0
        #frames_since_ping += 1
        #if frames_since_ping<frames_per_ping:
        #try:
        #    distance = ((bus.read_byte_data(0x70, 0x02) << 8) + bus.read_byte_data(0x70, 0x03)) / 100 # Get ranging result back
        #    bus.write_byte_data(0x70, 0x00, 0x51) # Request a new ranging
        #except Exception as e:
        #    pass
        #distance = 0
        #frames_since_ping = 0
    else:
        print("Vehicle not on the road!")

    sender.send(apex, distance)
    image_sender.send(image_with_apex)
