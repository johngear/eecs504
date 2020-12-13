import tensorflow as tf
import cv2
import numpy as np
import time

#tensorflow needs to be 1.13

def custom_cnn(data, saved_model_path):
    
    ## Set up the session by loading in the saved .pb file and other
    sess = tf.compat.v1.Session()
    tf.saved_model.load(sess, ['serve'], saved_model_path )
    
    inp = sess.graph.get_tensor_by_name('input_images:0')
    boxes_tensor = sess.graph.get_tensor_by_name('boxes:0')
    scores_tensor = sess.graph.get_tensor_by_name('scores:0')
    
    fbb = []
    
    ## Start of the process per image worth timing
    start = time.process_time()
    
    # Loop over each image
    for i in range(0,len(data)):

        # Read image in using opencv image methods. Same as Tommy's
        image = cv2.imread(data[i][0], cv2.IMREAD_COLOR)
        im = np.array(image).astype(np.float32)
        
        #print(im.shape)

        # Run the NN 
        boxes, scores = sess.run([boxes_tensor,scores_tensor], feed_dict={inp: np.expand_dims(im, axis=0)})
        
        #print(scores.shape)
        _,x = scores.shape
        ## need to reshape the data so it is consistent with the other methods
        scores = np.reshape(scores, (1,x,1))
        universal = np.concatenate((scores, boxes),2)
        universal = np.reshape(universal, (1,1,x,5))
        empty = np.zeros((1,1,x,2)) 
        universal = np.concatenate((empty, universal),3)
        
        ystart = universal[0][0][0][:][3]
        xstart = universal[0][0][0][:][4]
        yend = universal[0][0][0][:][5]
        xend = universal[0][0][0][:][6]
        
        universal[0][0][0][:][3] = xstart
        universal[0][0][0][:][4] = ystart 
        universal[0][0][0][:][5] = xend
        universal[0][0][0][:][6] = yend
        
        fbb.append(universal)

    return fbb
