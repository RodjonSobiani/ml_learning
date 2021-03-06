import cv2 as cv
import sys

print(cv.__version__)

img = cv.imread("../../opencv/assets/starry_night.jpg")
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("../../opencv/assets/starry_night.png", img)
    print("Image saved.")
elif k == ord("q"):
    cv.destroyAllWindows()
    sys.exit(0)
