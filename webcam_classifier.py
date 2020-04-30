import cv2
import numpy as np
from model import *
from multiprocessing import Pipe
import progressbar
from integral_image import *
import time

if __name__ == '__main__':
    model_shape = (19, 19) # dataset size
    model = AdaBoostModel.load('test_run_no_pad')
    cam = cv2.VideoCapture(0)
    # cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        # frame = cv2.imread('classmates.png')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        int_img = calc_int(gray)
        # print(frame.shape)
        # bar = progressbar.ProgressBar()
        faces_found = 0
        start = time.time()
        # for y in range(gray.shape[0]-model_shape[0]):
        #     for x in range(gray.shape[1]-model_shape[1]):
        #         # pred = model.classify(int_img[y:y+model_shape[0], x:x+model_shape[1]], recalc=False)
        #         # # faces_found += pred
        #         # # print('Found {} faces'.format(faces_found))
        #         # if pred == 1:
        #         #     # draw a window
        #         #     cv2.rectangle(frame, (x, y), (x+model_shape[1], y+model_shape[0]), (0, 255, 0))
        #         pass
        stop = time.time()
        # print(stop - start)
        # def classify_and_draw(gray, frame, y, x):
        #     pred = model.classify(gray[y:y+model_shape[1], x:x+model_shape[0]])
        #         # faces_found += pred
        #         # print('Found {} faces'.format(faces_found))
        #     if pred == 1:
        #         # draw a window
        #         cv2.rectangle(frame, (x, y), (x+model_shape[0], y+model_shape[0]), (0, 255, 0))
        
        # Pool(4).apply(classify_and_draw, np.ndenumerate(gray), {'gray': gray, 'frame': frame})
        # # # cv2.putText(frame, "Found {} faces".format(int(faces_found)), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow("Faces", frame)
        # cv2.waitKey()
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    cam.release()
    cv2.destroyAllWindows()