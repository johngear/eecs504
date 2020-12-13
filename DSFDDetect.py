import face_detection
import numpy as np

def DSFDDetect(img):
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    return detector.batched_detect(img)