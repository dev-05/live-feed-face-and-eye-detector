# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:54:18 2020

@author: dev
"""

import numpy as np
import cv2 as cv

face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.XML')
eye_cascade=cv.CascadeClassifier('haarcascade_eye.XML')

cap=cv.VideoCapture(0)

while(1):
    ret,img=cap.read()
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y), (x+w,y+h), (255,0,0) ,3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        
        for(xe,ye,we,he) in eyes:
            cv.rectangle(roi_color, (xe,ye) , (xe+we,ye+he),(0,255,0),2 )
    
    cv.imshow('img',img) 
    
    k=cv.waitKey(30)
    if k==13:
        break

cap.release()
cv.destroyAllWindows()