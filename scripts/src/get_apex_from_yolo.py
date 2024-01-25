from skimage import io, draw
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from  ultralytics import YOLO
import os
import sys




def detect_lanes(img_path, model):
    predictions = model.predict(source=img_path, save=False, conf=0.5)
    if predictions[0].masks is None:
        return None
    
    masks = predictions[0].masks.data
    cls = predictions[0].boxes.cls
    
    return (masks, cls)


def apex_from_mask(mask):
    height, width = mask.shape

    # Calculate the middle row index
    middle_row = height // 2

   # Initialize variables to track the indices of -1 values
    first_neg_one_idx = None
    last_neg_one_idx = None

    # Iterate over the rows from the middle row to the bottom row
    for y in range(middle_row, height):
        row = mask[y, :]
        # Check if there are any -1 values in the row
        if 1 in row:
            # If this is the first row with -1 values, record the indices and the line index
            if first_neg_one_idx is None:
                first_neg_one_idx = np.where(row == 1)[0][0]
                last_neg_one_idx = np.where(row == 1)[0][-1]
                line_idx = y
                center_idx = (first_neg_one_idx + last_neg_one_idx) // 2
                return (line_idx, center_idx )
      
    for y in range(0, middle_row):
        row = mask[y, :]
        # Check if there are any -1 values in the row
        if 1 in row:
            # If this is the first row with -1 values, record the indices and the line index
            if first_neg_one_idx is None:
                first_neg_one_idx = np.where(row == 1)[0][0]
                last_neg_one_idx = np.where(row == 1)[0][-1]
                line_idx = y
                center_idx = (first_neg_one_idx + last_neg_one_idx) // 2
                return (line_idx, center_idx )

    return None



def get_resized_coordinates(point, original_shape, new_shape):
    x, y = point
    h_original, w_original = original_shape
    h_resized, w_resized = new_shape
    scale_factor_x = float(w_resized) / w_original
    scale_factor_y = float(h_resized) / h_original
    x_resized = round(x * scale_factor_x)
    y_resized = round(y * scale_factor_y)
    return (x_resized, y_resized)



def apex_from_lanes(masks=None, cls=None, orginial_size=(720,1280)):
    # Auxiliary parameters
    focus_mask = None

    # Breask if no lanes have been found
    if masks is None:
        return None
    
    
    # If an element of cls is equal to 1, return the corresponding mask
    for i, elt in enumerate(cls):
        if elt== 1:
            focus_mask = masks[i]
    
    if focus_mask is None:
        focus_mask = masks[0]
    
    
    apex = apex_from_mask(focus_mask)
    
    if apex is None:
        return None
    
    x,y = apex
    x,y = get_resized_coordinates((x,y), original_shape=focus_mask.shape, new_shape=orginial_size)
    
    return x,y



def draw_circle(img,x,y,r=5):
    x = int(x)
    y = int(y)
    """
    Draw a circle of radius r on the passed img 
    on coordinates (x,y)
    """
    mask = np.zeros_like(img[:, :, 0])
    rr, cc = draw.disk((x, y), r, shape=mask.shape)
    mask[rr, cc] = 1

    # Apply the mask to the image
    img[mask == 1, :] = [255, 0, 0]  # Set the circle color to red
    
    return img


def get_apex(model, img_path):
    res = detect_lanes(img_path, model)
    if res is None:
        h,w = plt.imread(img_path).shape()
        return h,w
    masks, cls = res
    img = plt.imread(img_path)
    apex = apex_from_lanes(masks, cls, img.shape[:2])
    
    return apex