import cv2
import os
import sys

input_video = sys.argv[1]
output_folder = sys.argv[2]

# Open the video file
video = cv2.VideoCapture(input_video)

# Initialize a frame counter
frame_count = 0

os.makedirs(output_folder, exist_ok=True)

# Loop through the video frames
while True:
    # Read the next frame
    ret, frame = video.read()

    # If there are no more frames, break out of the loop
    if not ret:
        break
    
    if frame_count%1 == 0:
        # Save the frame as an image
        cv2.imwrite(f'{output_folder}/scene{frame_count}.jpg', frame)

    # Increment the frame counter
    frame_count += 1

# Release the video file
video.release()
