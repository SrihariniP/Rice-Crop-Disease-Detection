from math import copysign, log10

import numpy as np
from array import *
import os
import cv2
import csv
from PIL import Image
def resizeImage(img, target_w):
    img_h, img_w, d = img.shape
    image = img
    img_ratio = img_h/img_w
    target_h = target_w * img_ratio
    target_size = (round(target_w),round(target_h))
    print(target_size)
    resized_img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    #cv2.imshow("resized image", resized_img)
    return resized_img

def BackgroundElimination(imgMatrix):
    hsvImage = cv2.cvtColor(imgMatrix, cv2.COLOR_BGR2HSV)
    retval, thresh_gray = cv2.threshold(hsvImage, thresh=71, maxval=255, type=cv2.THRESH_BINARY)
    hsvMask = (thresh_gray[:, :, 1] > 250)  # & (thresh_gray[:,:,1]==0) & (thresh_gray[:,:,2]==0)
    imageNew = imgMatrix.copy()
    imageNew[:, :, 0] = imageNew[:, :, 0] * hsvMask
    imageNew[:, :, 1] = imageNew[:, :, 1] * hsvMask
    imageNew[:, :, 2] = imageNew[:, :, 2] * hsvMask
    return imageNew

def HealthyLeafElimination(imgMatrix):
    hsvImage1 = cv2.cvtColor(imgMatrix, cv2.COLOR_BGR2HSV)
    hueImage = hsvImage1[:, :, 1]
    retval1, thresh_gray1 = cv2.threshold(hsvImage1, thresh=22, maxval=255, type=cv2.THRESH_BINARY_INV)
    hsvMask1 = (thresh_gray1[:, :, 0] > 250)

    imageNew1 = imgMatrix.copy()
    imageNew1[:, :, 0] = imageNew1[:, :, 0] * hsvMask1
    imageNew1[:, :, 1] = imageNew1[:, :, 1] * hsvMask1
    imageNew1[:, :, 2] = imageNew1[:, :, 2] * hsvMask1

    return imageNew1

def cvt_infected_white(imgmatrix):
    hsv = cv2.cvtColor(imgmatrix, cv2.COLOR_BGR2HSV)
    retval1, thresh_gray1 = cv2.threshold(hsv, thresh=22, maxval=255, type=cv2.THRESH_BINARY_INV)
    Lower_hsv = np.array([250, 0, 0])
    Upper_hsv = np.array([255, 0, 0])

    Mask = cv2.inRange(thresh_gray1, Lower_hsv, Upper_hsv)

    return Mask

def cal_Hu_Moments(imgmatrix):
    moments = cv2.moments(imgmatrix)
    huMoments = cv2.HuMoments(moments)
    print(huMoments)
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments


def writeCSV(areas):
    f = open('E:/Sriharini/Rice_Crop_Disease_Detection/trial.csv', 'a', newline ='', encoding='UTF8')

    with f:
        # create the csv writer
        writer = csv.writer(f)

        for i in areas:
            # write a row to the csv file
            writer.writerow(i)

data = []
path = ('Rice_leaf_diseases/Trial')
#print("path = ",path)
for image in os.listdir(path):
    #Extracting Image
    img_path = path + "/" + image
    img = cv2.imread(img_path)
    original_img_name = "original" + " " + image
    cv2.imshow(original_img_name, img)

    #Resizing Image
    resized_img = resizeImage(img, 512)
    #cv2.imshow("resized image", resized_img)

    #Background Elimination
    bge_img = BackgroundElimination(resized_img)
    #cv2.imshow("bge image", bge_img)

    #Elimination of healthy portion of leaf
    le_img = HealthyLeafElimination(bge_img)
    preprocess_img_name = "le image" + " " + image
    cv2.imshow(preprocess_img_name, le_img)

    #Convertion of infected portion to white
    cvt_img = cvt_infected_white(resized_img)
    name = "cvt thresh grey" + " " + image
    cv2.imshow(name, cvt_img)

    #Calculating Hu Moments
    huMoments = cal_Hu_Moments(cvt_img)
    print("Hu Moments = ", huMoments)

    #Mean of RGB channels
    #channels = cv2.mean(le_img)
    mean_channels, std_channels =cv2.meanStdDev(le_img, mask = None)
    mean_channels = mean_channels.tolist()
    std_channels = std_channels.tolist()
    lst = []
    lst.append(mean_channels)
    lst.append(std_channels)
    data.append(lst)

writeCSV(data)

cv2.waitKey(0)
cv2.destroyAllWindows()