import cv2
from time import sleep
import numpy as np 
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import os
import sys
import pickle

camera = cv2.VideoCapture(0)

def get_image():
    retval, im = camera.read()
    return im

def scan_face(dirName):    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = []
    while len(faces) < 30:
        frame = get_image()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangle = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        if isinstance(rectangle, tuple):
            continue
        else:
            faces.append(rectangle[0])
        sleep(0.1)
        print(str(len(faces)) + "/30")

    count = 1
    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        fileName = dirName + "/" + str(count) + ".jpg"
        cv2.imwrite(fileName, roiGray)
        cv2.imshow("face", roiGray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1

# start the camera - take 30 frames, as the first ones are dark
for i in range(30):
    temp = camera.read() 
 
name = input("New user's id:\n")
dirName = "./users/" + name

if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Show your face to the camera and please wait")
    scan_face(dirName)
    print("Face scanning complete. Please, place your tag on the sensor")
    reader = SimpleMFRC522()
    try:
        reader.write(name)
    finally:
        GPIO.cleanup()
    print("User created")
else:
    print("User already exists")
    sys.exit()
del(camera)

# TRAIN CLASSIFIER
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "users") # change basedir to user dir

currentId = 1 # count how many users we have, so that classes will be encoded as 1,2,3..
labelIds = {}
yLabels = []
xTrain = []

for root, dirs, files in os.walk(imageDir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file) # find path to some photo
			label = os.path.basename(root) # find its folder name - label
			if not label in labelIds:
				labelIds[label] = currentId
				currentId += 1
			id_ = labelIds[label] # find photograph encoded class
			imageArray = cv2.imread(path,0) # read the image
			faces = face_cascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5) # again find face
			for (x, y, w, h) in faces:
				roi = imageArray[y:y+h, x:x+w]
				xTrain.append(roi) 
				yLabels.append(id_)


# save labels dictionary
with open("labels", "wb") as f:
	pickle.dump(labelIds, f)
	f.close()

recognizer.train(xTrain, np.array(yLabels)) # train recognizer
recognizer.save("trainer.yml") # save trained model
