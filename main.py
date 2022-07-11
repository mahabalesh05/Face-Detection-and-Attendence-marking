import numpy as np
import cv2
import face_recognition

img_elon1 = face_recognition.load_image_file('C:/Users/User/Desktop/FaceDetection_Attendence/AttendenceImages/elon2.jpg')
# Image is read in the form of 3D array
img_elon1 = cv2.cvtColor(img_elon1, cv2.COLOR_BGR2RGB)
# Syntax: cv2.cvtColor(src, code)
# Parameters:
# src: It is the image whose color space is to be changed.
# code: It is the color space conversion code.

img_elon2 = face_recognition.load_image_file('C:/Users/User/Desktop/FaceDetection_Attendence/AttendenceImages/elon3.jpg')
img_elon2 = cv2.cvtColor(img_elon2, cv2.COLOR_BGR2RGB)

img_bill = face_recognition.load_image_file('C:/Users/User/Desktop/FaceDetection_Attendence/AttendenceImages/Bill_Gates_speaks.jpg')
img_bill = cv2.cvtColor(img_bill, cv2.COLOR_BGR2RGB)

# 128 measurements are done to identify a face, encoding changes pic into array of 128 values.
encode_elon = face_recognition.face_encodings(img_elon1)[0]

face_loc = face_recognition.face_locations(img_elon1)[0]

# Forming a rectangle using pos of corners
cv2.rectangle(img_elon1, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255,0,250),2)
print(face_loc)

face_loc2 = face_recognition.face_locations(img_elon2)[0]
encode_elon2 = face_recognition.face_encodings(img_elon2)[0]
cv2.rectangle(img_elon2, (face_loc2[3], face_loc2[0]), (face_loc2[1], face_loc2[2]), (255,0,250),2)
print(face_loc2)

face_loc3 = face_recognition.face_locations(img_bill)[0]
encode_bill = face_recognition.face_encodings(img_bill)[0]
cv2.rectangle(img_bill, (face_loc3[3], face_loc3[0]), (face_loc3[1], face_loc3[2]), (255,5,250),2)
print(face_loc3)

results = face_recognition.compare_faces([encode_elon], encode_elon2)

# facedist reduces if the faces are similar and increases if faces are dissimilar
facedist = face_recognition.face_distance([encode_elon], encode_elon2)
print(results, facedist)

cv2.putText(img_elon2, f'{results} {round(facedist[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (150,0,100), 2)

cv2.imshow('Elon Musk', img_elon1)
cv2.imshow('Elon Test', img_elon2)
cv2.imshow('Bill test', img_bill)

cv2.waitKey(0)






