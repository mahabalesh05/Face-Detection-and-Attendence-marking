# Abstract of the Project:

# read all the face images in the file and add it to a list
# OpenCV was used to find encodings of the available images and add them to a list
# Web-cam was accessed using VideoCapture function
#


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'C:/Users/User/Desktop/FaceDetection_Attendence/AttendenceImages'
images = []
classnames = []
mylist = os.listdir(path)

print(mylist)

for cl in mylist:

    curr_image = cv2.imread(f'C:/Users/User/Desktop/FaceDetection_Attendence/AttendenceImages/{cl}' )
    images.append(curr_image)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)
print(images)

def MarkAttendence(name):
    with open('C:/Users/User/Desktop/FaceDetection_Attendence/Include/Attendence.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtstring}')


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = find_encodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # position and scale is mentioned to compress image
    imgsmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgsmall = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2RGB)

    face_loc = face_recognition.face_locations(imgsmall)
    encodescurr_frame = face_recognition.face_encodings(imgsmall,face_loc )

    # iterate through all faces and it is compared with all the encodings found
    for encodeface, faceloc in zip(encodescurr_frame, face_loc):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface )
        face_dist = face_recognition.face_distance(encodeListKnown, encodeface)
        print(face_dist)
        match_index = np.argmin(face_dist)

        if matches[match_index]:
            name = classnames[match_index].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendence(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


