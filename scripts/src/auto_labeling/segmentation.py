import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw
from tqdm import tqdm
from skimage.segmentation import find_boundaries
from skimage.segmentation import felzenszwalb

"""
All elements required to collect histograms to feed a binary classifier to detect orange segments.
Created by Romain Froger April 10, 2023.
"""


def x_y_from_img(s):
    x, y, _ = s.split('_')
    return (int(x), int(y))


def draw_circle(img,x,y,r=3):
    """
    Draw a circle of radius r on the passed img 
    on coordinates (x,y)
    """
    mask = np.zeros_like(img[:, :, 0])
    rr, cc = draw.circle(y, x, r, shape=mask.shape)
    mask[rr, cc] = 1

    # Apply the mask to the image
    img[mask == 1, :] = [0, 255, 0]  # Set the circle color to red
    
    return img


def plot_RGB_hist(arr):
    """
    Plot the RGB histogram given a one array hist.
    """
    
    sub_arrays = np.array_split(arr, 3)
    r = sub_arrays[0]
    g = sub_arrays[1]
    b = sub_arrays[2]
    # Plot the histograms
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].bar(range(len(r)), r)
    ax[1].bar(range(len(g)), g)
    ax[2].bar(range(len(b)), b)
    ax[0].set_title('Red Histogram')
    ax[1].set_title('Green Histogram')
    ax[2].set_title('Blue Histogram')
    plt.show()

    
def get_segment_histogram(img, segments, segment_id, n_bins=15):
    """
    Compute the histogram for the segment of one image
    n_bins allows the user to change the number of bins
    """
    
    # Mask for the segment
    mask = segments == segment_id
    # Extract pixels for the segment
    pixels = img[mask]
    
    
    lower_bound = 0
    upper_bound = 255
    bin_edges = np.linspace(lower_bound, upper_bound, n_bins+1)

    # Compute histogram on RGBs channel
    r_hist,_ = np.histogram(pixels[:, 0], bins=bin_edges, density=True)
    g_hist,_ = np.histogram(pixels[:, 1], bins=bin_edges, density=True)
    b_hist,_ = np.histogram(pixels[:, 2], bins=bin_edges, density=True)
    
    return np.concatenate([r_hist, g_hist, b_hist])


def get_connected_segments(segments, segment_label):
    """
    Given the segments of an image, return the list
    of connexe segments to segment_label
    """


    # Find the boundaries of the current segment
    current_segment = segments == segment_label
    segment_boundaries = find_boundaries(current_segment, mode="outer")

    # Find the labels of the connected segments
    connected_segments = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                # Skip the current segment
                continue
            y, x = segment_boundaries.nonzero()
            y += i
            x += j
            mask = (y >= 0) & (x >= 0) & (y < segments.shape[0]) & (x < segments.shape[1])
            labels = segments[y[mask], x[mask]]
            connected_segments += list(set(labels) - {segment_label})

    # Return the list of connected segments
    return list(set(connected_segments))

def propagate_segment(img, segments, init_coords, thr = 0.04):
    """
    Given an image, its segments and an initial coordinate (y,x),
    aggregates similar segments one to another. Similarity is computed
    using the euclidian_distance between two histograms.
    """
    hist_euclidian_threshold = thr
    y,x = init_coords
    # Mask for the segment
    mask = segments == segments[x][y]
    # Set considered segment to label -1
    segments[mask] = -1
    agg_hist = get_segment_histogram(img, segments, -1)
    
    #List containing the histograms of all aggregated segments
    hists = [agg_hist]
    
    while get_connected_segments(segments, -1):
        agg_count = 0
        #Loop through connexe segments
        for cur_seg_id in get_connected_segments(segments, -1):
            cur_hist = get_segment_histogram(img, segments, cur_seg_id)
            d = np.linalg.norm(agg_hist - cur_hist)
            if d < hist_euclidian_threshold:
                hists.append(cur_hist)
                #Add current segment to the -1 segment
                segments[segments == cur_seg_id] = -1
                #Recompute -1 segment histogram
                agg_hist = get_segment_histogram(img, segments, -1)
                agg_count += 1
        if agg_count == 0:
            break
            
    return segments, hists


def preview_lane(img, segment, apex):
    """
    Preview the detected lane given the aggregated segments
    """
    line_mask = segment == -1
    output_img = img.copy()
    output_img[line_mask] = [0, 255, 0]  # orange color
    output_img = draw_circle(output_img, apex[0],apex[1])
    
    io.imshow(output_img)
    
    
    
def apex_from_segments(segments):
    height, width = segments.shape

    # Calculate the middle row index
    middle_row = height // 2

   # Initialize variables to track the indices of -1 values
    first_neg_one_idx = None
    last_neg_one_idx = None

    # Iterate over the rows from the middle row to the bottom row
    for y in range(middle_row, height):
        row = segments[y, :]
        # Check if there are any -1 values in the row
        if -1 in row:
            # If this is the first row with -1 values, record the indices and the line index
            if first_neg_one_idx is None:
                first_neg_one_idx = np.where(row == -1)[0][0]
                last_neg_one_idx = np.where(row == -1)[0][-1]
                line_idx = y
                center_idx = (first_neg_one_idx + last_neg_one_idx) // 2
                return (line_idx, center_idx)
            
    for y in range(0, middle_row):
        row = segments[y, :]
        # Check if there are any -1 values in the row
        if -1 in row:
            # If this is the first row with -1 values, record the indices and the line index
            if first_neg_one_idx is None:
                first_neg_one_idx = np.where(row == -1)[0][0]
                last_neg_one_idx = np.where(row == -1)[0][-1]
                line_idx = y
                center_idx = (first_neg_one_idx + last_neg_one_idx) // 2
                return (line_idx, center_idx)

    return -1



def auto_label_img(img_path, clf):
    img = plt.imread(img_path)
    segments = felzenszwalb(img, scale=20, sigma=1, min_size=50)

    for seg_idx in np.unique(segments):
        hist = get_segment_histogram(img, segments, seg_idx) 
        y_pred = clf.predict(np.array(hist).reshape(1,-1))
        if y_pred == 1:
            segments[segments == seg_idx] = -1
            
    prediction = apex_from_segments(segments)
    
    if prediction != -1:
        return prediction


    
def predict_apex(img_folder, clf):
    image_paths = os.listdir(img_folder)
    
    for filename in tqdm(image_paths):
        if filename.endswith(".jpg"):
            apex = auto_label_img(os.path.join(img_folder, filename), clf)

            if apex is not None:
                new_name = f"%d_%d_%s" % (apex[0], apex[1], filename)
                # Use the rename() method to rename the file
                os.rename(os.path.join(img_folder, filename), os.path.join(img_folder, new_name))
    
    print("Apex prediction done for current folder")
    
