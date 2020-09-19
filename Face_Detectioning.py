# Importing libraries
from PIL import Image
import inspect
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Creating the classifier
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Opening the Image
image = cv.imread('lotr.jpg')
# Converting the color spaces in order to analyse correctly
fixed_image = cv.cvtColor(image,cv.COLOR_BGR2RGB) 

# Detecting the Faces
faces = face_classifier.detectMultiScale(image,1.3,5)

# If there is no face on the photo, print an info message
if faces is():
    print('No faces found')

# Function to detect the face 
def detect_face(fixed_image):
    face_rects = face_classifier.detectMultiScale(fixed_image)
    for(x,y,w,h) in face_rects:
        cv.rectangle(fixed_image, (x,y), (x+w, y+h), (255,0,0), 10)
    return fixed_image

result = detect_face(fixed_image)
plt.imshow(result)
plt.show()
