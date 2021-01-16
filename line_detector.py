import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm


def detect_lines(image_path, lower=185, upper=240):

    lower = np.uint8([lower])
    upper = np.uint8([upper])

    im_dataset_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # im_dataset_gray = cv2.GaussianBlur(im_dataset_gray,(3, 3),0)
    im_dataset_gray = cv2.resize(im_dataset_gray, (180, 180))
    
    blur = cv2.GaussianBlur(im_dataset_gray,(3, 3),0)
    # blur = im_dataset_gray #test
    canny = cv2.Canny(blur, 30, 100)
    range = cv2.inRange(im_dataset_gray, lower, upper)

    blur_canny = cv2.GaussianBlur(canny,(3, 3),0)
    blur_range = cv2.GaussianBlur(range,(5, 5),0)
    bitwise = cv2.bitwise_and(blur_canny, blur_canny, mask=blur_range)

    # plt.imshow(blur)
    # plt.show()

    # plt.imshow(blur_canny)
    # plt.show()

    # plt.imshow(blur_range)
    # plt.show()

    # plt.imshow(bitwise)
    # plt.show()

    # plt.imshow(im_dataset_gray)
    # plt.show()


    df = pd.DataFrame(bitwise).apply(lambda  x: (x/x)*255)
    bitwise = np.uint8(df)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 70  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 45  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = bitwise.copy()  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(bitwise, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255),0.1)
    except:
        pass

    # plt.imshow(line_image)

    return line_image

datadir_read = 'original_pictures'

datadir_write = 'detected_lines'

directions = ["w", "a", "d"]


for direction in directions:
    curr_datadir_read = os.path.join(datadir_read, direction)
    curr_datadir_write = os.path.join(datadir_write, direction)

    print (curr_datadir_read)
    print (curr_datadir_write)

    for img in tqdm(os.listdir(curr_datadir_read)):

        a = detect_lines(os.path.join(curr_datadir_read, img), 170, 240)
        cv2.imwrite(os.path.join(curr_datadir_write, img), a)
    

