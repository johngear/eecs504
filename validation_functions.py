import cv2
from matplotlib import pyplot as plt
import numpy as np

def get_data(group, path):
    '''
    Creates a list of tuples, each with an image name and its bounding box statistics.
    Input:
        group: what picture group we want, e.g. couples
        path: path to the wider_faces_train_bbx_gt.txt
    Output:
        data: list of tuples of data (filename, bbs (list itself))
    '''
    # Based on https://towardsdatascience.com/how-do-you-train-a-face-detection-model-a60330f15fd5

    f = open(path, 'r')
    file = f.readlines()
    data = []
    count = 0

    for i in file:
        # Grab the files for specific group
        if group in i:

            # image name and remove \n
            filename = file[count]
            filename = filename[:-1]

            # get bounding boxes
            numFaces = int(file[count + 1])
            bbsStart = count + 2
            boxes = []

            occ = 0
            pose = 0
            ill = 0
            for j in range(0, numFaces):
                face = file[bbsStart + j]
                bbs = face.split(' ')
                ill = max(ill, int(bbs[6]))
                occ = max(occ, int(bbs[8]))
                pose = max(pose, int(bbs[9]))
                bbs = bbs[:4]
                bbs = [int(k) for k in bbs]
                boxes.append(bbs)

            # Create a tuple with file name and list of bounding boxes, and save
            if occ != 0 or pose != 0 or ill != 0:
                pass
            else:
                temp = (filename, boxes)
                data.append(temp)

        count += 1
    return data


def min_dist(x, y, boxes):
    '''
    Finding the nearest bounding box (if not in order)
    Input:
        x: detected starting x
        y: detected starting y
        boxes: list of all the bounding boxes for the image
    '''
    mindist = 1e10  # arbitrary large number

    for k in range(0, len(boxes)):
        x1 = boxes[k][0]
        y1 = boxes[k][1]
        dist = np.sqrt((x1 - x) ** 2 + (y1 - x) ** 2)  # Euclidean distance
        if dist < mindist:
            mindist = dist
            mink = k

    return mink


def get_iou(bb1, bb2):
    """
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Or: bb1 is a tuple with data (x1, y1, x2, y2)
        bb2 is a tuple with data (x1, y1, x2, y2)

    Returns
    -------
    float
        in [0, 1]
    """
    # determine the coordinates of the intersection rectangle

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def plot_results(result):
    acc = []
    for i in range(0, len(result)):
        acc.extend(result[i][1])

    acc_arr = np.array(acc)
    avg_acc = np.mean(acc_arr)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    idx = 0
    for i in range(0, len(result)):
        ar = result[i][1]
        plt.scatter(range(idx, idx + len(ar)), ar)
        idx = idx + len(ar)

    ax.set_title('Average of IoU: {}'.format(avg_acc))
    ax.set_ylabel('Intersection over Unions Average')
    ax.axes.xaxis.set_visible(False)
    plt.show()

    bad = [result[i][2] - result[i][3] for i in range(0, len(result))]

    n, bins, patches = plt.hist(x=bad, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Incorrect')
    plt.ylabel('Frequency')
    plt.title('Number of Misdetections (NumCorrect-NumDetected)')
    plt.show()


