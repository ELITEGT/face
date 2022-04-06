# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:23:41 2021

@author: Tian, Gama, Rio
"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hidung_deteksi = cv2.CascadeClassifier('Nariz.xml')
cap = cv2.VideoCapture(0)
masker_on = False

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 2)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        if masker_on:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Masker ada', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'Masker tidak ada', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        hidung = hidung_deteksi.detectMultiScale(roi_gray, 1.18, 20)

        for (sx, sy, sw, sh) in hidung:
            cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (255, 0, 0), 1)
            cv2.putText(img, 'Hidung', (x + sx, y + sy), 1, 1, (255, 255, 255), 1)

        if len(hidung) > 0:
            masker_on = False
        else:
            masker_on = True

    cv2.putText(img, 'Jumlah Wajah : ' + str(len(faces)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('deteksi wajah', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()