import cv2
import numpy as np


## Converting image from one format to another

# image = cv2.imread('image.jpg', flags = cv2.IMREAD_COLOR) # default converting with BGR
# image_grey = cv2.imread('image.jpg', flags = cv2.IMREAD_GRAYSCALE) # converting to gray
# cv2.imwrite('MyPic.png', image)
# cv2.imwrite('GreyPic.jpg', image_grey)
# cv2.imshow('test_image', image) #show an image in a window
# cv2.waitKey()
# cv2.destroyAllWindows()

## There are many different flags for imread

## Capturing from a live camera

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)
print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()
cv2.destroyWindow('MyWindow')
cameraCapture.release()

