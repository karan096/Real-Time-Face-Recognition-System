import os
import scipy.misc
import cv2
import sys
import shutil
import random
import numpy as np
import pylab
import shelve

cur_dir = os.path.dirname(__file__)
faces_dir = os.path.join(cur_dir, "att_faces")


content = [line.rstrip('\n') for line in open('names.txt')]

dir_list=next(os.walk(faces_dir))[1]
no_of_faces = len(dir_list)
no_of_train_img = 10

total_training_images = no_of_faces * no_of_train_img
img_path = os.path.join(faces_dir, "s1/1.pgm")
img = scipy.misc.imread(img_path)
size = img.shape
rows = size[0]
colums = size[1]

cnt_img = 0
image_matrix = np.empty(shape=(rows * colums, total_training_images), dtype='float64')
for face_no in xrange(1, no_of_faces + 1):
	for train_img_no in xrange(1, no_of_train_img + 1):
		img_path = os.path.join(faces_dir, 's' + str(face_no), str(train_img_no) + '.pgm')
		img_2d = cv2.imread(img_path, 0)
		img_1d = np.array(img_2d, dtype='float64').flatten()
		image_matrix[:, cnt_img] = img_1d
		cnt_img += 1
mean_1d_img = np.sum(image_matrix, axis=1) / total_training_images
mean_2d_img = mean_1d_img.reshape(rows,colums)

for i in xrange(0, total_training_images):
	image_matrix[:, i] -= mean_1d_img[:]
covariance_matrix = np.matrix(image_matrix.transpose()) * np.matrix(image_matrix)
covariance_matrix /= total_training_images

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
sort_indices = eigen_values.argsort()[::-1]
eigen_values = eigen_values[sort_indices]                              
eigen_vectors = eigen_vectors[sort_indices]                            
eigen_vectors = eigen_vectors.transpose()                              
eigen_vectors = image_matrix * eigen_vectors                           
norms = np.linalg.norm(eigen_vectors, axis=0)                          
eigen_vectors = eigen_vectors / norms                                  
weights_matrix = eigen_vectors.transpose() * image_matrix

#def recognize_img(img_path):
def recognize_img(img_2d):
	#img_2d = cv2.imread(img_path, 0)
	img_1d = np.array(img_2d, dtype='float64').flatten()                     
	img_1d -= mean_1d_img
	img_1d = np.reshape(img_1d, (rows*colums, 1))                            
	S = eigen_vectors.transpose() * img_1d                                
	diff = weights_matrix - S
	norms = np.linalg.norm(diff, axis=0)
	closest_face_id = np.argmin(norms)
	return (closest_face_id / no_of_train_img) + 1

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detectFaceOnline():
    # access the webcam (every webcam has a number, the default is 0)
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # to detect faces in video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y-30),(x+w,y+h+30),(255,0,0),2)
            roi_gray = gray[y-30:y+h+30, x:x+w]
            try:
                res = cv2.resize(roi_gray,(92, 112))
                text_color = (128,0,128)
                detected_face_id = recognize_img(res)
                cv2.putText(frame, "Hello "+content[detected_face_id-1] + " !!" , (0,50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1.8, text_color, thickness=2)
            except:
                continue
        cv2.imshow('Real Time Face Recognition',frame)
        k=cv2.waitKey(30) &0xff
        if k==27:
			break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


detectFaceOnline()
