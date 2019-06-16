# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys

# Declare input argument parser:
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=0,required=True)
args = parser.parse_args()

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

##############################
## Declare requirement inputs:
##############################
# Path to your model:
MODEL_NAME = os.path.join("trained_models","grab_model")
CWD_PATH = os.getcwd()
# Path to your inference graph:
PATH_TO_INFERENCE_MODEL = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to your label_map:
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'label_map.pbtxt')
# Number of classes:
NUM_CLASSES = 196
# Path to your video input:
PATH_TO_VIDEO = os.path.join("test_videos",args.input)

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

##############################################
## Load the Tensorflow model into new session:
##############################################
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_INFERENCE_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

######################################################################
## Initiate output tensors from the model (boxes,class,scores,num):
######################################################################
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rates = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera input:
camera = cv2.VideoCapture(PATH_TO_VIDEO)
ret = camera.set(3,1280)
ret = camera.set(4,720)

################################################
## Start classification and video playback:
################################################
while(True):
    # Initiate clock (fps):
    t_start = cv2.getTickCount()

    # Read video frames into single column RBG array using cv:
    ret, frame = camera.read()
    frame_expanded = np.expand_dims(frame, axis=0)

   # Perform classification based on inference graph and generate correspoding output (boxes,scores,classes,num):
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Using Tensorflow Object Detection API to draw the frame rectangle in real-time:
    # Lower down the min_score_thresh, if its difficult to show any output:
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.7)

    # Show FPS in video_playback:
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rates),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    
    # Images compiled with draw overlay and ready for display:
    cv2.imshow('Grab Car Classifier', frame)

    t_end = cv2.getTickCount()
    time_1 = (t_end-t_start)/freq
    frame_rates = 1/time_1

    # Press 'q' to quit program:
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()

cv2.destroyAllWindows()

