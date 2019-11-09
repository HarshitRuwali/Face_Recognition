#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

#cap = cv2.VideoCapture(0)

img_1='/Users/harshitruwali/Pictures/ProfilePhoto.jpeg'

cap = cv2.imread(img_1)



#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#or
#face_cascade = cv2.CascadeClassifier('/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalcatface_extended.xml')

'''
while 1:
	img = cap.read(img_1)
    #img = cv2.imread(img_1)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.load('haarcascade_frontalface_default.xml')
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
	      
'''

while 1:
	img = cap.read(img_1)
	faces = face_cascade.load('haarcascade_frontalface_default.xml')

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

