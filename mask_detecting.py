import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
import cv2 as cv

IMG_SIZE = 150
model = load_model('Mask_detector.h5')
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Error')
        break

    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h+50, x:x+w]
        roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
        img = cv.resize(roi, (IMG_SIZE, IMG_SIZE))
        data = img
        data = np.expand_dims(data, axis=0)
        result = model.predict(data)
        if np.argmax(result) == 0:
            cv.rectangle(frame, (x,y), (x+w,y+h+50), (0,0,255), 1)
        elif np.argmax(result) == 1:
            cv.rectangle(frame, (x,y), (x+w,y+h+50), (0,255,255), 1)
        elif np.argmax(result) == 2:
            cv.rectangle(frame, (x,y), (x+w,y+h+50), (0,255,0), 1)

        cv.imshow('Frame', frame)

        if cv.waitKey(1) == ord('q'):
            exit()

cap.release()
cv.destroyAllWindows()
