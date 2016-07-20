# CS6475-FP
CS6475 Final Project

VIDEO.py is run first. It takes a video texture, matches a template and using the template, creates a mask from the video. The final process is the blending of the video texture with a background scene. Set CreateMaskWithMotion = True for the intiial run. If only blending is required, set CreateMaskWithMotion = False.

AUTO_SIZE.py is used to for feature matching based on a template. The program will automatically generate a properly sized mask based on the template. It will also locate the feature to replace and build a second image of the correct shape (=mask.shape, =image1.shape) to facilitate blending.

RESIZE.py is used to crop and size images. The sizing must be known, and entered as vars into the program.

MATCHER.py was an experiment, and was not used for this project. It was an attempt to use ORB and SIFT for feature detection. However, I was not able to get this to produce the desired results. Template matching was more effective for this project.
