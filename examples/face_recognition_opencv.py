from cv2.cv2 import CascadeClassifier, imread, rectangle, imshow, waitKey, destroyAllWindows

# load the pre-trained model
classifier = CascadeClassifier('../utilities/haarcascade_frontalface_default.xml')


# Show function
def showBoxesOnImage(faces, image):
    # print bounding box for each detected face
    for box in faces:
        print(box)
    # print bounding box for each detected face
    for box in faces:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)
    # show the image
    imshow('face detection', image)
    # keep the window open until we press a key
    waitKey(0)
    # close the window
    destroyAllWindows()


# ---------------------------First image---------------------------
# load the photograph
pixels = imread('../images/test1.jpg')
# perform face detection
boxes = classifier.detectMultiScale(pixels)
showBoxesOnImage(boxes, pixels)
# -----------------------------------------------------------------

# ---------------------------Second image---------------------------
# load the photograph
pixels = imread('../images/test2.jpg')
# perform face detection
boxes = classifier.detectMultiScale(pixels, 1.05, 8)
showBoxesOnImage(boxes, pixels)
# -----------------------------------------------------------------

# ---------------------------Third image---------------------------
# load the photograph
pixels = imread('../images/test3.jpg')
# perform face detection
boxes = classifier.detectMultiScale(pixels, 1.02, 6)
showBoxesOnImage(boxes, pixels)
# -----------------------------------------------------------------
