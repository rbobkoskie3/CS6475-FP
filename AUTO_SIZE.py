# FINAL
# Robert Bobkoskie
# rbobkoskie3

import os, re, sys
import cv2
import numpy as np
import scipy as sp
import scipy.signal


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

def ResizeImage(img, img2_shape):

    #print 'SIZE', img.shape, img2_shape
    resize_img = cv2.resize(img, (img2_shape[1], img2_shape[0]))

    return resize_img

def CreateMask(img, template, idx):
    '''
    This function uses Template Matching Functions from opencv to create a mack for each frame:
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    '''
    
    mask = np.zeros(shape=img.shape, dtype=np.float64)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = img.astype('float32')
    #template = template.astype('float32')

    #print 'IMG SHAPE', img.shape, img.dtype
    #print 'TEMPLATE SHAPE', template.shape, template.dtype
    #img = img[100:img.shape[0]-100, 400:img.shape[1]-500]
    #print 'CROP SIZE', img.shape

    d, w, h = template.shape[::-1]
    top_left = (0, 0)
    #w = template.shape[1]
    #h = template.shape[0]

    method = eval('cv2.TM_CCOEFF')
    method = eval('cv2.TM_CCOEFF_NORMED')
    method = eval('cv2.TM_CCORR')
    method = eval('cv2.TM_SQDIFF')
    method = eval('cv2.TM_SQDIFF_NORMED')
    method = eval('cv2.TM_CCORR_NORMED')

    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val >= 0.80: #Use 0.90 for most matches, 0.80 is needed to match the moving car in all frames

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        # FOR VERIFICATION - Draw rectangle around Car
        #bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv2.rectangle(img, top_left, bottom_right,(200,200,200), 2)
        #cv2.imwrite(str(idx).zfill(4)+'TEST.jpg', img)  #FOR VERIFYING DETECTED SHAPE

        # Create mask
        mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = 255.0

        # Create Image to Blend
        bld_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
        #bld_img = img[top_left[1]:top_left[1] + h - 10, top_left[0]:top_left[0] + w - 10]  #FOR TESTING
        #cv2.imshow('IMAGE', bld_img)
        #cv2.waitKey()

    return mask, top_left, bld_img


def main():

    mypath_inp = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\INPUT'
    mypath_msk = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\MASK\\'
    mypath_bld = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\BLEND'
    outdir     = 'C:\PYCODE\CS6475\PROJECT FINAL\FACE RECOGNITION\PROJECT\FACES\RESULTS\\'

    files_inp = [os.path.join(mypath_inp, f) for f in os.listdir(mypath_inp) if os.path.isfile(os.path.join(mypath_inp, f))]
    files_msk = [os.path.join(mypath_msk, f) for f in os.listdir(mypath_msk) if os.path.isfile(os.path.join(mypath_msk, f))]
    files_bld = [os.path.join(mypath_bld, f) for f in os.listdir(mypath_bld) if os.path.isfile(os.path.join(mypath_bld, f))] 
    #template = cv2.imread('template.jpg',0) # SINGLE CHANNEL
    template = cv2.imread('template.jpg')
    #print 'TEMPLATE SHAPE', template.shape
    #cv2.imshow('IMAGE', template)
    #cv2.waitKey()

    # Create a mask from a moving object
    for idx, (inp_name, bld_name) in enumerate(zip(files_inp, files_bld)):
    #for idx, img_name in enumerate(files_inp):
        img = cv2.imread(inp_name)      #FOR DYNAMIC SCENE
        #img = cv2.imread('street.jpg')  #FOR STATIC SCENE
        blend_img = cv2.imread(bld_name)
        #img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        mask, top_left, bld_img = CreateMask(img, template, idx)

        #print outdir+str(idx).zfill(4)+'.jpg'
        cv2.imwrite(mypath_msk+str(idx).zfill(4)+'.jpg', mask) #FOR VERIFICATION

        bld_mask = np.zeros(shape=img.shape, dtype=np.float64)
        resize_img = ResizeImage(blend_img, template.shape)
        if top_left != (0, 0):

            ##################################
            #print bld_mask.shape, resize_img.shape, top_left[1], top_left[0]
            #bld_mask[top_left[1]:top_left[1] + resize_img.shape[0], top_left[0]:top_left[0] + resize_img.shape[1]] = resize_img
            bld_mask[top_left[1]:top_left[1] + resize_img.shape[0], top_left[0]:top_left[0] + resize_img.shape[1]] = bld_img  #USE FOR FINAL BLEND
            #bld_mask[top_left[1]:top_left[1] + resize_img.shape[0] - 10, top_left[0]:top_left[0] + resize_img.shape[1] - 10] = bld_img  #USE FOR FINAL BLEND - TESTING
            ##################################

        #cv2.imwrite(outdir+str(idx).zfill(4)+'.jpg', bld_mask) #FOR VERIFICATION
        #print bld_mask.shape, mask.shape, img.shape
        #cv2.imshow('G IMAGE', blendImage)
        #cv2.waitKey()
        #print outdir+str(idx).zfill(4)+'.jpg'

        ##################################
        #final_Image = BlendImagePair(img, bld_mask, mask)
        final_Image = BlendImagePair(blend_img, bld_mask, mask)  #USE FOR FINAL BLEND
        ##################################

        cv2.imwrite(outdir+str(idx).zfill(4)+'.jpg', final_Image)

if __name__ == '__main__':
    main()

