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

def process_frame(q, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    int_img = np.cumsum(np.cumsum(gray, axis=0), axis=1)
    # print(frame.shape)
    # bar = progressbar.ProgressBar()
    faces_found = 0
    start = time.time()
    for y in range(gray.shape[0]-model_shape[0]):
        for x in range(gray.shape[1]-model_shape[1]):
            pred = model.classify(int_img[y:y+model_shape[0], x:x+model_shape[1]], recalc=False)
            # faces_found += pred
            # print('Found {} faces'.format(faces_found))
            if pred == 1:
                # draw a window
                cv2.rectangle(frame, (x, y), (x+model_shape[1], y+model_shape[0]), (0, 255, 0))
            # pass
    q.put(frame)

if __name__ == '__main__':
    model_shape = (19, 19) # dataset size
    model = AdaBoostModel.load('test_run_cascade')
    cam = cv2.VideoCapture(0)
    # cv2.namedWindow("test")
    # q = Queue(-1)
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if frame is None:
            continue
        orig_shape = (frame.shape[1], frame.shape[0])
        # frame = cv2.imread('classmates.png')
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scale_percent = 25
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100) 
        dim = (160, 120)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        int_img = np.cumsum(np.cumsum(gray, axis=0), axis=1)
        # # print(frame.shape)
        # # bar = progressbar.ProgressBar()
        faces_found = 0
        start = time.time()
        max_zoom = 2.0
        step = 1.3
        zoom = 1.0
        tmp = resized
        while zoom < max_zoom:
            for y in range(tmp.shape[0]-model_shape[0]):
                for x in range(tmp.shape[1]-model_shape[1]):
                    pred = model.classify(int_img[y:y+model_shape[0], x:x+model_shape[1]], recalc=False)
                    # faces_found += pred
                    # print('Found {} faces'.format(faces_found))
                    if pred == 1:
                        # draw a window
                        cv2.rectangle(resized, (int(x*zoom), int(y*zoom)), (int((x+model_shape[1])*zoom), int((y+model_shape[0])*zoom)), (0, 255, 0))
                    # pass
            zoom *= step
            tmp = cv2.resize(tmp, (int(tmp.shape[1]/zoom), int(tmp.shape[0]/zoom)), interpolation=cv2.INTER_AREA)

        stop = time.time()
        print(stop - start)
        # def classify_and_draw(gray, frame, y, x):
        #     pred = model.classify(gray[y:y+model_shape[1], x:x+model_shape[0]])
        #         # faces_found += pred
        #         # print('Found {} faces'.format(faces_found))
        #     if pred == 1:
        #         # draw a window
        #         cv2.rectangle(frame, (x, y), (x+model_shape[0], y+model_shape[0]), (0, 255, 0))
        
        # Pool(4).apply(classify_and_draw, np.ndenumerate(gray), {'gray': gray, 'frame': frame})
        # # # cv2.putText(frame, "Found {} faces".format(int(faces_found)), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        new_frame = cv2.resize(resized, orig_shape, interpolation=cv2.INTER_AREA)
        cv2.imshow("Faces", new_frame)
        # cv2.waitKey()
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if k%256 == ord('s'):
            dirpath = os.path.abspath('./saves')
            filename = 'capture_at_{}.jpg'.format(time.strftime("%Y-%m-%d-%H-%M-%S"))
            filepath = os.path.join(dirpath, filename)
            path = pathlib.Path(filepath)
            Image.fromarray(new_frame.astype(np.uint8)).save(path, format="JPEG")
            print(f"Saved at {filename}")
    cam.release()
    cv2.destroyAllWindows()