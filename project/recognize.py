import cv2
import numpy as np 
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import pickle
from time import sleep
import os
import sys

def get_image():
    retval, im = camera.read()
    return im

with open('labels', 'rb') as f: # load dictionary with labels
	dict = pickle.load(f)
	f.close()

print("Please place your tag on the sensor")
reader = SimpleMFRC522()
try:
    id, text = reader.read()
    recognized = False
    for name, value in dict.items():
        if int(name) == int(text):
            recognized = True
            face_id = value
    if recognized == False:
        print("This tag is not valid.")
        sys.exit()

finally:
        GPIO.cleanup()

print("Wait, your face is being scanned")
sleep(1)
camera = cv2.VideoCapture(0)

# start the camera - take 30 frames, as the first ones are dark
for i in range(30):
    temp = camera.read()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

font = cv2.FONT_HERSHEY_SIMPLEX

img = get_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rectangle = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

while isinstance(rectangle, tuple): # make sure that the face is in the photo
    img = get_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangle = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

for (x, y, w, h) in rectangle:
    roiGray = gray[y:y+h, x:x+w]
    id_, conf = recognizer.predict(roiGray)
    if int(id_) != int(face_id) or conf < 40:
        print("Face not recognized! Access denied")
        number = len(os.listdir('./intruders'))
        cv2.imwrite("intruders/intruder" + str(number) + ".jpg", img)
        sys.exit()
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Access successful", (x, y), font, 2, (0, 0 ,255), 2,cv2.LINE_AA)

cv2.imshow('frame', img)
key = cv2.waitKey(0)

cv2.destroyAllWindows()