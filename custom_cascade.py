def custom_cascade(data):
    clf = CascadeClassifier.load('cascade')
    fbb = []
    faces = []
    for i in range(0,len(data)):
        image = (cv2.imread(data[i][0], cv2.IMREAD_GRAYSCALE))
        print(i)
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
            faces = []
            faces_mod = np.zeros((faces.shape[0], 8))
            faces_mod[:, 2:3] = np.ones((faces.shape[0], 1))
            faces_mod[:, 3:7] = faces
            faces_mod = np.reshape(faces_mod, (1, 1, faces_mod.shape[0], faces_mod.shape[1]))
        
        fbb.append(faces_mod)
    return fbb
