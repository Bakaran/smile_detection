# imports
import cv2 as cv
import numpy as np
from yunnet import YuNet
# imports


class Model:
    # constructor function with variables TrainX,TrainY,TestX,TestY,X,Y,Path,Model
    def __init__(self):
        (self.TrainX,
         self.TrainY,
         self.TestX,
         self.TestY,
         self.X,
         self.Y,
         self.Path,
         self.model) = ([], [], [], [], [], [], "genki4k/files", None)

    # crop image function
    def cropImage(self, image):
        # convert to grayscale of each frames
        grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # read the haarcascade to detect the faces in an image
        faceCascade = cv.CascadeClassifier(
            'haarcascade_frontalface_alt.xml')

        # detects faces in the input image
        faces = faceCascade.detectMultiScale(grayImg, 1.25, 5)
        # loop over all detected faces
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # To draw a rectangle in a face
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
                Face = image[y:y+h, x:x+w]
                return Face
        else:
            return image
