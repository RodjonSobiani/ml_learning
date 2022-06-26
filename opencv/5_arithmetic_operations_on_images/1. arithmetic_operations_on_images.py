import numpy as np
import cv2 as cv
import sys

img1 = cv.imread('../../opencv/assets/ml.png')
img2 = cv.imread('../../opencv/assets/opencv-logo.png')

# if img1 is None:
#     sys.exit("Could not read the image.")
# cv.imshow("Display window", img1)
# k = cv.waitKey(0)
# if k == ord("q"):
#     cv.destroyAllWindows()
#     sys.exit(0)

if img2 is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img2)
k = cv.waitKey(0)
if k == ord("q"):
    cv.destroyAllWindows()
    sys.exit(0)

dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
