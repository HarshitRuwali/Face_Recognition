'''

import face_recognition
import cv2


image = face_recognition.load_image_file("53895966.jpg")
face_locations = face_recognition.face_locations(image)


image = face_recognition.load_image_file("53895966.jpg")
face_locations = face_recognition.face_locations('/Users/harshitruwali/Pictures')


img='/Users/harshitruwali/Pictures/ProfilePhoto.jpeg'

i=cv2.imread(img,0)
i2=cv2.imread(img)
img = cv2.imread('img',0)


cv2.imshow('image', i)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

import numpy as np
import cv2


#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#face_cascade.load('haarcascade_frontalface_default.xml')
#face_cascade =cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')


#cap = cv2.VideoCapture(0)
img_1='/Users/harshitruwali/Pictures/ProfilePhoto.jpeg'

har = cv2.imread(img_1)

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalcatface_extended.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while 1:
    #ret, img = cap.read()
    img_1 = cv2.imread(har)
    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img_1,(x,y),(x+w,y+h),[225],2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_1[y:y+h, x:x+w]

        if (NULL):
            print("null error")
        
        '''
       	eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
       
     	    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
     	'''
    '''
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    '''

cv2.imshow('img',img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cap.release()
cv2.destroyAllWindows()
'''


