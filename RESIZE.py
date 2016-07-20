# ASSIGNMENT 4
# Robert Bobkoskie
# rbobkoskie3

import os, re
import cv2
import numpy as np
import scipy as sp


def ResizeImg(image, img):

    #n = 32
    #n = 128
    #n = 256
    n = 512
    #n = 1024
    #n = 2048

    r = n * 1.0 / image.shape[1]
    dim = (n, int(image.shape[0] * r))

    print 'ORIG SIZE', img, image.shape
    #if re.search(r'IMG_6550-XL.jpg', img):
        #resize_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        #resize_img = cv2.resize(image, (1944, 1296))
        #print resize_img.shape
        #cv2.imwrite(img, resize_img)

    #resize_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #resize_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)

    # Slice 53 rows from bottom of image
    #resize_img = image[0:image.shape[0] - 53,:]
    #Slice rows from bottom and columns from right side of image
    #resize_img = image[0:image.shape[0] - 10, 0:image.shape[1] - 10]
    #resize_img = image[:, 0:image.shape[1] - 5]
    #resize_img = image[:, 0+10:image.shape[1]]

    resize_img = np.zeros((288, 512, 3), dtype=np.uint8)
    
    #'''
    for i in range(3):
        #print resize_img[..., i][128:resize_img.shape[0]-121, 55:resize_img.shape[1]-425].shape, image[..., i].shape
        resize_img[..., i][128:resize_img.shape[0]-121, 55:resize_img.shape[1]-425] = image[..., i]
        #cv2.imshow('G IMAGE', resize_img)
        #cv2.waitKey()
    #'''

    print 'RESIZE', img, resize_img.shape, '\n'
    cv2.imwrite(img, resize_img)

def main():

    #mypath = 'C:\PYCODE\CS6475\PROJECT 7\WRITEUp'
    #mypath = 'C:\PYCODE\CS6475\PROJECT 7\IMAGES'
    mypath = 'C:\PYCODE\CS6475\RESIZE'
    print 'PATH', mypath
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for img in files:
        #print img
        image = cv2.imread(img)
        ResizeImg(image, img)

if __name__ == '__main__':
    main()

