import numpy as np
import cv2+
import time
import pylab
import os

#import the cascade for face detection

cur_dir = os.path.dirname(__file__)
faces_dir = os.path.join(cur_dir, "att_faces")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fo = open("names.txt", "a")


def TakeSnapshotAndSave(dir_name):
    # access the webcam (every webcam has a number, the default is 0)
    person_name = raw_input("Enter your name: ");
    fo.write( person_name +"\n");
    current_sample_dir = os.path.join(faces_dir, dir_name)
    if not os.path.exists(current_sample_dir):
        os.makedirs(current_sample_dir)
    cap = cv2.VideoCapture(0)
    count = 1
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # to detect faces in video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        global roi_gray
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y-30),(x+w,y+h+30),(255,0,0),2)
            roi_gray = gray[y-30:y+h+30, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        x = 0
        y = 20
        text_color = (0,255,0)
        # write on the live stream video
        cv2.putText(frame, "Press q when ready   " + str(count) + ": " , (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=2)


        # if you want to convert it to gray uncomment and display gray not fame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        # press the letter "q" to save the picture

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # write the captured image with this name
            img_path = os.path.join(current_sample_dir, str(count) + ".pgm")
            res = cv2.resize(roi_gray,(92, 112))
            cv2.imwrite(img_path,res)
            count = count + 1
            if count==11:
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


dir_list=next(os.walk(faces_dir))[1]
sample_num=len(dir_list)+1
TakeSnapshotAndSave("s"+str(sample_num))
