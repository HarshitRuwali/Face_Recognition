#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2


# In[ ]:


cap = cv2.VideoCapture(0)
# or
#img_1='/Users/harshitruwali/Pictures/ProfilePhoto.jpeg'

cap = cv2.imread(img_1)


# In[ ]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#or
#face_cascade = cv2.CascadeClassifier('/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalcatface_extended.xml')


# In[ ]:


while 1:
    ret, img = cap.read()
    #or 
    #img = cv2.imread(img_1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        


# In[ ]:


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

