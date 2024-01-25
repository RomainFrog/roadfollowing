#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import sys

device = torch.device('gpu')

model = model.to(device)
model_path = sys.argv[1]
output_name = sys.argv[2]

model.load_state_dict(torch.load(model_path))


from torch2trt import torch2trt

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)

torch.save(model_trt.state_dict(), output_name)

