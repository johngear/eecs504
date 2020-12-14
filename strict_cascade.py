import numpy as np
import cv2


def strict_cascade(data):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    fbb = []
    
    for i in range(0,len(data)):
        image = cv2.imread(data[i][0], cv2.IMREAD_GRAYSCALE)

        faces = face_cascade.detectMultiScale(image, 1.1, 4) 
        eyes = eye_cascade.detectMultiScale(image, 1.1, 4) 
        
        
        if len(faces) ==0 :
            faces_mod = np.zeros((1,8))
            faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
        else: 
            nfaces = []
            for face in faces:
                flag = 0
                for eye in eyes:
                    if eye[0] > face[0] and eye[1] < face[1] and eye[2] > face[2] and eye[3] < face[3]:
                        flag = 1
                if flag == 1:
                    nfaces.append(face)
            if len(nfaces) == 0:
                faces_mod = np.zeros((1,8))
                faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
            else:
                nfaces = np.ndarray(nfaces)
                faces_mod = np.zeros((nfaces.shape[0], 8))
                faces_mod[:, 2:3] = np.ones((nfaces.shape[0], 1))
                faces_mod[:, 3:7] = nfaces
                faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
        
        fbb.append(faces_mod)

    return fbb