import cv2
import numpy as np
from model import *
from multiprocessing import Pipe
import progressbar
from integral_image import *
import time
import datetime
from multiprocessing.queues import Queue
from PIL import Image
import os
import pathlib
from util import PickleMixin

if __name__ == '__main__':
    # Specify the model file to be used and camera number
    model_file = 'cascade_50-100'
    camera_number = 0

    # Define the model shape we trained with
    model_shape = (19, 19) # dataset size

    #Define both the model and the camera to capture and define from
    model = PickleMixin.load(model_file)
    cam = cv2.VideoCapture(camera_number)

    # Loop through taking camera input
    while True:
        #Read the frame and determine success of reading
        ret, frame = cam.read()

        # If there is a camera error, we break
        if not ret:
            break

        if frame is None:
            continue
        
        # Determine the image shape python is reading in from the webcam
        orig_shape = (frame.shape[1], frame.shape[0])
        
        # New dimensions for the image (width, height)
        dim = (160, 120)
        
        # Lower the size for the image (to increase computations)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Create a grayscale image (RGB colors -> GRAY)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate our integral image:
        int_img = calc_int(gray)

        # Initiate a timer point to calculate time between images
        start = time.time()
        
        # Run through our classification:
        # Define the range of feature size regions (zoom = multiplied factor with feature of size 19 pixels)
        zoom = 1.0
        max_zoom = 1.0
        step = 1.3

        # Temporary resized image:
        tmp = resized
        
        while zoom <= max_zoom:
            #Loop through each possible feature location starting from top left:
            for y in range(tmp.shape[0]-model_shape[0]):
                for x in range(tmp.shape[1]-model_shape[1]):
                    # Get our prediction for the model based on the target region
                    pred = model.classify(int_img[y:y+model_shape[0], x:x+model_shape[1]], recalc=False)
                    if pred == 1:
                        # If we predict a feature, draw a box around the region (RGB)
                        color = (0, 255, 0)

                        cv2.rectangle(resized, (int(x*zoom), int(y*zoom)), (int((x+model_shape[1])*zoom), int((y+model_shape[0])*zoom)), color)
                    # Else, do nothing   
            zoom *= step
            tmp = cv2.resize(tmp, (int(tmp.shape[1]/zoom), int(tmp.shape[0]/zoom)), interpolation=cv2.INTER_AREA)
        
        # End the computation time for each detection
        stop = time.time()
        print(stop - start)

        # Resize the image back up to the original size
        new_frame = cv2.resize(resized, orig_shape, interpolation=cv2.INTER_AREA)
        
        # Display the image
        cv2.imshow("Faces", new_frame)

        # Wait for a keypress and determine the key
        k = cv2.waitKey(1)

        # If the key pressed during the 1 millesecond is the escape key:
        if k%256 == 27:
            print("Escape hit, closing...")
            break

        # If we want to save, we can capture the previous image
        if k%256 == ord('s'):
            dirpath = os.path.abspath('./')
            filename = 'capture_at_{}.jpg'.format(time.strftime("%Y-%m-%d-%H-%M-%S"))
            filepath = os.path.join(dirpath, filename)
            path = pathlib.Path(filepath)
            Image.fromarray(new_frame.astype(np.uint8)).save(path, format="JPEG")
            print(f"Saved at {filename}")

    # Cleanup resources:
    cam.release()
    cv2.destroyAllWindows()