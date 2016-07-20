import sys, os
import numpy as np
import scipy as sp
import scipy.signal
import cv2

#KNN_MATCH = True
KNN_MATCH = False

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                                 % cv2.__version__)

# This magic line appears to be needed with OpenCV3 to prevent the feature
# detector from throwing an error...
cv2.ocl.setUseOpenCL(False)


def findMatchesBetweenImages(img1, img2):
    """
    This function detects and computes SIFT (or ORB) from the input images, and
    returns the best matches using the normalized Hamming Distance.
    """

    img1 = cv2.blur(img1,(2, 2))
    img1 = cv2.Canny(img1,100, 200)

    img2 = cv2.blur(img2,(2, 2))
    img2 = cv2.Canny(img2,100, 200)

    #img1 = cv2.Sobel(img1,cv2.CV_8U, 1,0, ksize=5)
    #img2 = cv2.Sobel(img2,cv2.CV_8U, 1,0, ksize=5)
    

    # matches - type: list of cv2.DMath
    matches = None
    # image_1_kp - type: list of cv2.KeyPoint items.
    kp1 = None
    # image_1_desc - type: numpy.ndarray of numpy.uint8 values.
    desc1 = None
    # image_2_kp - type: list of cv2.KeyPoint items.
    kp2 = None
    # image_2_desc - type: numpy.ndarray of numpy.uint8 values.
    desc2 = None

    # Code modified from:
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # Code also pulled from 'feature_matching.zip'
    # print 'DTYPE', image_1.dtype

    # Initialize ORB detector object
    orb = SIFT()  # or cv2.SIFT() in OpenCV 2.4.9+

    # Find keypoints, compute descriptors and show them on original image (with scale and orientation)
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    print "Image 1: {} keypoints found".format(len(kp1))
    print "Image 2: {} keypoints found".format(len(kp2))


    if KNN_MATCH:
        #####################################
        # FLANN parameters
        desc1 = desc1.astype('float32')
        desc2 = desc2.astype('float32')
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        all_matches = flann.knnMatch(desc1, desc2, k=2)

        matches = []
        for m,n in all_matches:
            if m.distance < 0.75*n.distance:
                matches.append([m])
        #print matches
        #####################################

    else:
        #####################################
        # Create BFMatcher (Brute Force Matcher) object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors
        matches = bf.match(desc1, desc2)
        print "{} matches found".format(len(matches))

        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)
        matches = matches[:10]

        print 'DMatch.distance - Distance between descriptors. The lower, the better it is:'
        for x in matches:
            print x.distance,
        print '\n'
        #####################################

    return kp1, kp2, matches


if __name__ == "__main__":

    sourcefolder = os.path.abspath(os.path.join(os.curdir, 'images', 'source'))
    outfolder = os.path.abspath(os.path.join(os.curdir, 'images', 'output'))

    print 'Image source folder: {}'.format(sourcefolder)
    print 'Image output folder: {}'.format(outfolder)
    print 'Searching for folders with images in {}.'.format(sourcefolder)

    # Extensions recognized by opencv
    exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
            '.jpe', '.jp2', '.tiff', '.tif', '.png']

    # For every image in the source directory
    for dirname, dirnames, filenames in os.walk(sourcefolder):

        image_1 = None
        image_2 = None

        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext.lower() in exts:
                if '_1' in name:
                    print "Reading image_1 {} from {}.".format(filename, dirname)
                    image_1 = cv2.imread(os.path.join(dirname, filename))

                elif '_2' in name:
                    print "Reading image_2 {} from {}.".format(filename, dirname)
                    image_2 = cv2.imread(os.path.join(dirname, filename))

        if image_1 == None or image_2 == None:
            print "Did not find image_1 / image_2 images in folder: " + dirname
            continue
        else:
            print "Found images in folder {}, processing them.".format(dirname)

        print "Computing matches."
        image_1_kp, image_2_kp, matches = findMatchesBetweenImages(
            image_1, image_2)

        print "Visualizing matches."
        img2 = np.zeros((1,1))
        if KNN_MATCH:
            output = cv2.drawMatchesKnn(image_1, image_1_kp, image_2, image_2_kp,
                                        matches, img2)
        else:
            output = cv2.drawMatches(image_1, image_1_kp, image_2, image_2_kp,
                                     matches, img2)

        print "Writing images to folder {}".format(outfolder)
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        cv2.imwrite(os.path.join(outfolder, "matches.jpg"), output)
