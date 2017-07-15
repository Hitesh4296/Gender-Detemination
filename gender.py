import cv2
import numpy as np
from keras.models import load_model

cam = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


# loading saved cnn model
model = load_model('gender_reco.h5')


# predicting face emotion using saved model
def get_gender(im):
    im = im.reshape((-1,50,50,1))
    gender=model.predict(im)
    if (gender[0][0==1]):
        gender = 'Female'
    else:
        gender= 'Male'
    return gender


# reshaping face image
def recognize_face(im):
    im = cv2.resize(im, (50, 50))
    return get_gender(im)

while True:
    ret, fr = cam.read()
    if ret == True:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            # Extract detected face
            fc = fr[y:y+h, x:x+w, :]
            # resize to a fixed shape
            r = cv2.resize(fr, (50, 50)).flatten()

            text = recognize_face(fr)
            cv2.putText(fr, text, (x, y), font, 1, (255, 255, 0), 2)

            cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('fr', fr)
        if cv2.waitKey(1) == 27:
            break
    else:
        print "error"
        break



cv2.destroyAllWindows()