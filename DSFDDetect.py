import face_detection
import numpy as np

def DSFDDetect(img):
    fbb = []
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    for i in range(0,len(data)):
        image = cv2.imread(data[i][0], cv2.IMREAD_GRAYSCALE)
        faces = detector.detect(img)
        faces = faces[:,0:4]
        if len(faces) ==0 :
            faces_mod = np.zeros((1,8))
            faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
        else: 
            faces_mod = np.zeros((faces.shape[0], 8))
            faces_mod[:, 2:3] = np.ones((faces.shape[0], 1))
            faces_mod[:, 3:7] = faces
            faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
        
        fbb.append(faces_mod)
    
    return fbb