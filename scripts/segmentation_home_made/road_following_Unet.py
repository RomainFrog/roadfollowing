#! /usr/bin/env python3

from jetcam.csi_camera import CSICamera
# from jetcam.usb_camera import USBCamera

camera = CSICamera(width=224, height=224, capture_fps=30)
# camera = USBCamera(width=224, height=224)


camera.running = True

from tensorflow import keras
import tensorflow as tf
import numpy as np

class MeanMetricWrapper(tf.keras.metrics.Mean):
    def __init__(self, name='mean_metric', dtype=None):
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)

    def get_config(self):
        return {'name': self.name}

    @classmethod
    def from_config(cls, config):
        return cls(name=config['name'])

model = tf.keras.models.load_model("first_full_model", custom_objects={'MeanMetricWrapper': MeanMetricWrapper})

print(tf.test.is_built_with_cuda())  # Vérifie si TensorFlow est construit avec CUDA
print(tf.config.list_physical_devices('GPU'))  # Récupère la liste des appareils GPU disponibles

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
from get_apex_from_yolo import * 


import numpy as np
from remote_control.control_sender import ControlSender
from remote_control.image_sender import ImageSender
from smbus import SMBus

sender = ControlSender("10.0.1.1", 1234)
image_sender = ImageSender("10.0.1.1", 1235)

distance = 0
frames_per_ping = 5
frames_since_ping = 0
print("MODEL LOADED WITH SUCCESS")

while True:
    image = cv2.flip(camera.value,-1)

    image = image*1./255
    img_dem = np.expand_dims(image, axis=0)
    
    res_dem = model.predict(img_dem)[0]
    orange_lines = compute_mask(res_dem[:,:,2])
    white_lines = compute_mask(res_dem[:,:,1])

    #img_res = img_dem[0].copy()
    #img_res[orange_lines != 0] = [1, 0, 0]
    #img_res[white_lines != 0] = [0, 1, 0]

    apex = apex_from_mask(orange_lines)
    if pred:
        prediction = cv2.circle(img_dem, apex[::-1], 3, (0, 0, 1), 3)
    else:
        apex = apex_from_mask(white_lines)
        prediction = cv2.circle(img_dem, apex[::-1], 3, (0, 0, 1), 3)

    y = apex[0]
    x = apex[1]

    #prediction_widget.value=bgr8_to_jpeg(prediction)
    apex = float(apex[0])
    distance = 0

    sender.send(apex, distance)
    image_sender.send(prediction)