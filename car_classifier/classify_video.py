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

##############################
## Declare requirement inputs:
##############################
# Path to your model:
MODEL_NAME = os.path.join("trained_models","grab_model")
CWD_PATH = os.getcwd()
# Path to your inference graph:
PATH_TO_INFERENCE_MODEL = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to your label_map:
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labels.txt')
# Number of classes:
NUM_CLASSES = 196
# Path to your video input:
PATH_TO_VIDEO = os.path.join("test_videos",args.input)

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

# Read available classes from label:
def read_label(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
    label_list = []
    for line in lines:
        label_list.append(line[:-1])
    return label_list

################################################
## Start classification and video playback:
################################################
# Declare the min threshold % for classification:
min_thres_classify = 0.5

while(True):
    # Initiate clock (fps):
    t_start = cv2.getTickCount()

    # Read video frames into single column RBG array using cv:
    ret, frame = camera.read()
    # Get shapes from images:
    frame_h, frame_w, channels = frame.shape
    frame_expanded = np.expand_dims(frame, axis=0)

   # Perform classification based on inference graph and generate correspoding output (boxes,scores,classes,num):
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Read labels to acquire all classes values:
    labels = read_label(PATH_TO_LABELS)

    # Tabulate parameters from prediction:
    for i in range(int(num)):
        # Boxes coordinate:
        top, left, bottom, right = boxes[0][i]
        # Classified classes, reason to deduct by 1 to make sure the index start from 0 instead of 1:
        cur_class = int(classes[0][i]) - 1
        # Current score %:
        cur_score = scores[0][i]
        
        # Make sure only output classes with confidence which more than 0.5:
        if cur_score > min_thres_classify:
          # Resize the coordinate based on the image size ratio:
          x_min = left * frame_w
          y_min = bottom * frame_h
          x_max = right * frame_w
          y_max = top * frame_h

          cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 2)
          output_class = str(labels[cur_class]) + " : " + str(cur_score)
          cv2.putText(frame,output_class, (int(x_min),int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))

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

