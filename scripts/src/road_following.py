#! /usr/bin/env python3

from jetcam.csi_camera import CSICamera
# from jetcam.usb_camera import USBCamera

camera = CSICamera(width=224, height=224, capture_fps=65)
# camera = USBCamera(width=224, height=224)

camera.running = True

import torch
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))

import ipywidgets
import traitlets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

def flip_and_jpeg(img):
    return bgr8_to_jpeg(cv2.flip(img, -1))

# create image preview
#prediction_widget = ipywidgets.Image(width=camera.width, height=camera.height)
#display(prediction_widget)

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
    prediction = image.copy()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = output[0]
    y = output[1]
        
    x = int(camera.width * (x / 2.0 + 0.5))
    y = int(camera.height * (y / 2.0 + 0.5))
    prediction = cv2.circle(prediction, (x, y), 8, (255, 0, 0), 3)
    #prediction_widget.value=bgr8_to_jpeg(prediction)
    apex = float(output[0])
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

    sender.send(apex, distance)
    image_sender.send(prediction)
