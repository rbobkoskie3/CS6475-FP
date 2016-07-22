# FINAL
# Robert Bobkoskie
# rbobkoskie3

import os, re, sys
import cv2
import numpy as np
import scipy as sp
import scipy.signal

#####################################
# Set CreateMaskWithMotion = True if generating a mask from a moving object
# else, CreateMaskWithMotion = False will use a static mask

CreateMaskWithMotion = True
#CreateMaskWithMotion = False
#####################################


def BlendImagePair(img, car, mask):

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #car = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    #cv2.imshow('G IMAGE', g_image)
    #cv2.waitKey()

    #####################################
    # START Code from assignment 6, blend image:
    #####################################
    def generatingKernel(parameter):
        kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                           0.25, 0.25 - parameter /2.0])

        return np.outer(kernel, kernel)

    def reduce(image):
        kernel = generatingKernel(0.4)
        conv_image = scipy.signal.convolve2d(image, kernel, 'same')

        return conv_image[::2, ::2].astype(np.float64) 

    def expand(image):
        kernel = generatingKernel(0.4)
        conv_image = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)   #create np.zeros
        conv_image[::2,::2] = image[:,:]    #assign every other [row, col] the value of the image [row, col]
        return 4*scipy.signal.convolve2d(conv_image, kernel, 'same')

    def gaussPyramid(image, levels):
        output = [image]

        for i in range(levels):
            reduce_image = reduce(image)
            reduce_image = reduce_image.astype(np.float64)
            #print reduce_image.dtype
            output.append(reduce_image)
            image = reduce_image

        return output

    def laplPyramid(gaussPyr):
        output = []

        for gaussPyr_ind in range(0, len(gaussPyr) - 1, 1):
            gaussian = gaussPyr[gaussPyr_ind]
            expGaussPyr = expand(gaussPyr[gaussPyr_ind + 1])

            if gaussian.shape[0] != expGaussPyr.shape[0]:
                expGaussPyr = np.delete(expGaussPyr, -1, axis=0)

            if gaussian.shape[1] != expGaussPyr.shape[1]:
                expGaussPyr = np.delete(expGaussPyr, -1, axis=1)

            laplacian = gaussian - expGaussPyr
            output.append(laplacian)

        output.append(gaussPyr[-1])
        return output

    def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
        blendedPyr = []

        for i in range(0, len(laplPyrWhite), 1):
            #print 'HERE --', laplPyrWhite[i].shape, laplPyrBlack[i].shape, gaussPyrMask[i].shape
            output = np.zeros(shape=laplPyrWhite[i].shape, dtype=np.float64)

            output = ( gaussPyrMask[i] * laplPyrWhite[i] +
                       (1 - gaussPyrMask[i]) * laplPyrBlack[i] )

            blendedPyr.append(output)

        return blendedPyr

    def collapse(pyramid):

        for ind in range(len(pyramid), 0, -1):

            if ind - 2 >= 0:
                if ind - len(pyramid) == 0:  #first pass
                    pyr_exp = expand(pyramid[ind - 1])
                else:
                    pyr_exp = expand(sum_pyr)

                pyr = pyramid[ind - 2]

                if pyr_exp.shape[0] != pyr.shape[0]:
                    pyr_exp = np.delete(pyr_exp, -1, axis=0)
                if pyr_exp.shape[1] != pyr.shape[1]:
                    pyr_exp = np.delete(pyr_exp, -1, axis=1)

                sum_pyr = pyr_exp + pyr

        return sum_pyr

    def run_blend(black_image, white_image, mask):
        # Automatically figure out the size
        #print 'HERE ---', black_image.shape, white_image.shape, mask.shape
        min_size = min(black_image.shape)
        #depth = int(math.floor(math.log(min_size, 2))) - 4   #use math, at least 16x16 at the highest level.
        depth = int(np.floor(np.log2(min_size))) - 4       #use np, at least 16x16 at the highest level.
        #print 'LAYERS ---', depth
        
        gauss_pyr_mask = gaussPyramid(mask, depth)
        gauss_pyr_black = gaussPyramid(black_image, depth)
        gauss_pyr_white = gaussPyramid(white_image, depth)


        lapl_pyr_black  = laplPyramid(gauss_pyr_black)
        lapl_pyr_white = laplPyramid(gauss_pyr_white)

        outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
        outimg = collapse(outpyr)

        # Accountfor blending that sometimes results in slightly out of bound numbers
        outimg[outimg < 0] = 0
        outimg[outimg > 255] = 255
        outimg = outimg.astype(np.uint8)

        return outimg

    #####################################
    # END Code from assignment 6, blend image:
    #####################################

    black_img = img.astype(float)
    white_img = car.astype(float)
    mask_img = mask.astype(float) / 255

    # outimg = run_blend(black_img, white_img, mask_img) #For a single grey channel
    outimg = np.zeros(black_img.shape, dtype=np.uint8)
    for i in range(3):
        outimg[..., i] = run_blend(black_img[..., i], white_img[..., i], mask_img[..., i])

    return outimg


def CreateMask(img, template, idx):
    '''
    This function uses Template Matching Functions from opencv to create a mack for each frame:
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    '''
    

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    print 'ORIG SIZE', img.shape
    #img = img[100:img.shape[0]-100, 400:img.shape[1]-500]
    #print 'CROP SIZE', img.shape

    w, h = template.shape[::-1]

    method = eval('cv2.TM_CCOEFF')
    method = eval('cv2.TM_CCOEFF_NORMED')
    #method = eval('cv2.TM_CCORR')
    method = eval('cv2.TM_CCORR_NORMED')
    method = eval('cv2.TM_SQDIFF')
    method = eval('cv2.TM_SQDIFF_NORMED')

    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # Draw rectangle around Car
    #bottom_right = (top_left[0] + w, top_left[1] + h)
    #cv2.rectangle(img, top_left, bottom_right, 255, 2)
    #cv2.imwrite(str(idx)+'TEST.jpg', img)

    # Create mask of car
    mask = np.zeros(shape=img.shape, dtype=np.float64)
    mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = 255.0

    return mask


    '''
    #ret,thresh = cv2.threshold(img,50,100,100)
    #ret,thresh = cv2.threshold(img,75,200,cv2.THRESH_TOZERO_INV)
    #thresh = cv2.adaptiveThreshold(img,10,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    #ret,thresh = cv2.threshold(img,30,10,cv2.THRESH_BINARY_INV)
    ret,thresh = cv2.threshold(img,200,10,cv2.THRESH_BINARY_INV)
    #contours = cv2.findContours(thresh, 1, 2)
    contours = cv2.findContours(thresh, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #print cnt.dtype
    #cnt = cnt.astype(np.float64)

    # Image moments 
    #M = cv2.moments(cnt)
    #print M

    x,y,w,h = cv2.boundingRect(cnt)
    #print x,y,w,h
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #img = img[y:y+h,x:x+w]
    print 'CROP SIZE', img.shape
    cv2.imwrite(str(idx)+'TEST.jpg', img)

    ############################
    # RESIZE IMAGE
    # Calculate the ratio of the new image to the old image
    w = 100.0
    c = w / img.shape[1]
    dim = (int(w), int(img.shape[0] * c))

    # Perform the actual resizing of the image
    resize_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print 'RE SIZE', resize_img.shape
    #cv2.imwrite('TEST.jpg', resize_img)
    ############################
    '''

    '''
    ############################
    # EDGE DETECTION: Sobel gradient filters or High-pass filters
    # Output dtype = cv2.CV_8U
    sobelx8u = cv2.Sobel(img,cv2.CV_8U, 1,0, ksize=5)
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img,cv2.CV_64F, 1,0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    #cv2.imwrite('TEST.jpg', sobel_8u)
    ############################

    ############################
    # EDGE DETECTION: Canny
    img = cv2.blur(img,(2, 2))
    edges = cv2.Canny(img,100, 200)
    #edges = cv2.Canny(img,200, 100)
    #cv2.imwrite('TEST.jpg', edges)
    ############################

    ############################
    # WORKS, BUT REMOVES ALL BLACK BORDERS, 0
    ret,thresh = cv2.threshold(img, 1,255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    cv2.imwrite('TEST.jpg', crop)
    ############################
    '''

def main():

    mypath_car = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\INPUT'
    outdir     = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\MASK\\'
    files = [os.path.join(mypath_car, f) for f in os.listdir(mypath_car) if os.path.isfile(os.path.join(mypath_car, f))]
    template = cv2.imread('template.jpg',0)
    #cv2.imshow('G IMAGE', template)
    #cv2.waitKey()

    if CreateMaskWithMotion:
        # Create a mask from a moving object
        for idx, img_name in enumerate(files):
            img = cv2.imread(img_name)
            #img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            mask = CreateMask(img, template, idx)
            #print outdir+str(idx).zfill(4)+'.jpg'
            cv2.imwrite(outdir+str(idx).zfill(4)+'.jpg', mask)

    mypath_msk = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\MASK'
    mypath_car = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\INPUT'
    outdir     = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\RESULTS\\'
    
    files_msk = [os.path.join(mypath_msk, f) for f in os.listdir(mypath_msk) if os.path.isfile(os.path.join(mypath_msk, f))]
    files_car = [os.path.join(mypath_car, f) for f in os.listdir(mypath_car) if os.path.isfile(os.path.join(mypath_car, f))]    

    if CreateMaskWithMotion:
        img = cv2.imread('C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\street.jpg')
    else:
        mask = cv2.imread('C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\MASK.jpg')

    for idx, (img_msk, img_car) in enumerate(zip(files_msk, files_car)):

        if CreateMaskWithMotion:
            mask = cv2.imread(img_msk)
        else:
            img = cv2.imread(img_msk)

        car = cv2.imread(img_car)
        blendImage = BlendImagePair(img, car, mask)
        #cv2.imshow('G IMAGE', blendImage)
        #cv2.waitKey()
        print outdir+str(idx).zfill(4)+'.jpg'
        cv2.imwrite(outdir+str(idx).zfill(4)+'.jpg', blendImage)

if __name__ == '__main__':
    main()

