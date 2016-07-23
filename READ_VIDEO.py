# FINAL
# Robert Bobkoskie
# rbobkoskie3

import os, re, sys
import cv2
import numpy as np
import scipy as sp


def Edit_Vid(vid, mypath, idx):

    #Code adapted from:
    #http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction
    #http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
    #http://www.fourcc.org/codecs.php
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter('output.avi',-1, 20.0, (512,288))
    out = cv2.VideoWriter('output.avi',fourcc, 24.0, (512,288))

    cap = cv2.VideoCapture(vid)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret==True:
            # NEED TO GET THE SHAPE OF THE FRAME
            # NOTE THAT THE SHAPE IS (ROWS, COLS), THE PARAMETER IN
            # cv2.VideoWriter('output.avi',fourcc, 24.0, (WIDTH,HEIGHT))
            # IS WIDTH X HEIGHT
            #print frame.shape

            #frame = cv2.flip(frame,0)
            fgmask = fgbg.apply(frame)

            # Write the frame
            out.write(frame)

            #'''
            # Play Video
            cv2.imshow('frame',fgmask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #'''

        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    '''
    # Play video
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''

    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('G IMAGE', g_image)
    #cv2.waitKey()
    filename = mypath+'\\OUTPUT\\'+idx+'.jpg'
    print 'ORIG SIZE', img.shape
    #img = img[100:img.shape[0]-100, 400:img.shape[1]-500]
    #print 'CROP SIZE', filename, img.shape
    #cv2.imwrite(filename, img)


    #ret,thresh = cv2.threshold(img,50,100,100)
    ret,thresh = cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV)
    #ret,thresh = cv2.threshold(img,75,200,cv2.THRESH_TOZERO_INV)
    #thresh = cv2.adaptiveThreshold(img,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

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
    img = img[y:y+h,x:x+w]
    print 'CROP SIZE', img.shape
    #cv2.imwrite(filename, img)

    ############################
    # RESIZE IMAGE
    # Calculate the ratio of the new image to the old image
    w = 100.0
    c = w / img.shape[1]
    dim = (int(w), int(img.shape[0] * c))

    # Perform the actual resizing of the image
    resize_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print 'RE SIZE', resize_img.shape
    cv2.imwrite(filename, resize_img)
    ############################
    '''

def main():

    mypath = 'C:\PYCODE\CS6475\PROJECT FINAL\VIDEO\SOURCE'
    #print 'PATH', mypath
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    # Crop and Resize Images
    for idx, vid in enumerate(files):
        Edit_Vid(vid, mypath, str(idx).zfill(4))

if __name__ == '__main__':
    main()

