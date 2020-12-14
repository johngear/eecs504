def custom_cascade(data):
    clf = CascadeClassifier.load('cascade')
    fbb = []
    faces = []
    for i in range(0,len(data)):
        image = (cv2.imread(data[i][0], cv2.IMREAD_GRAYSCALE))
        if image:
            M,N = np.shape(image)
            for i1 in range(M - 19):
                for i2 in range(N - 19):
                    if clf.classify(image[i1:i1+19,i2:i2+19]):
                        faces.append([i1,i2,i1+19,i2+19])

        
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
            faces_mod = np.zeros((nfaces.shape[0], 8))
            faces_mod[:, 2:3] = np.ones((nfaces.shape[0], 1))
            faces_mod[:, 3:7] = nfaces
            faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
        
        fbb.append(faces_mod)
    return fbb
