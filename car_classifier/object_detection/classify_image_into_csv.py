import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv

##############################
## Declare requirement inputs:
##############################
#Path to your test image:
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#Path to your model:
MODEL_NAME = 'trained_models\grab_model'
CWD_PATH = os.getcwd()
#Path to your inference graph:
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
#Path to your label_map:
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labels.txt')

##############################################
## Load the Tensorflow model into new session:
##############################################
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

######################################################################
## Get all the output tensors from the model (boxes,class,scores,num):
######################################################################
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

##########################
## Read input test images:
##########################
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
print('\nTotal test images: ' + str(len(TEST_IMAGE_PATHS)))
print('Start generating your test result...')

# Read available classes from label:
def read_label(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
    label_list = []
    for line in lines:
        label_list.append(line[:-1])
    return label_list

# Total of 7 columns are generated in CSV:
csv_content = [''] * 7
csv_header = ['filename','predict_1','confidence_1','predict_2','confidence_2','predict_3','confidence_3']

################################
## Execute image classification:
################################
with open("model_classify_result.csv",'w') as resultFile:
    # Initialize CSV writeups:
    wr = csv.writer(resultFile, delimiter=",", lineterminator="\n")
    wr.writerow(csv_header)
    for k in TEST_IMAGE_PATHS:
        top_3_classes = []
        # Read images into array using cv:
        frame = cv2.imread(k)
        # Get shapes from images:
        frame_h, frame_w, channels = frame.shape
        # Extend the dimension for prediction:
        frame_expanded = np.expand_dims(frame, axis=0)
        
        # Perform classification based on inference graph and generate correspoding output (boxes,scores,classes,num):
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Read labels to acquire all classes values:
        labels = read_label(PATH_TO_LABELS)
        # Retrieve classes and confidence level from prediction:
        for i in range(int(num)):
            # Classified classes, reason to deduct by 1 to make sure the index start from 0 instead of 1:
            cur_class = int(classes[0][i]) - 1 
             # Current score %:
            cur_score = scores[0][i]      
            # Collect only the top 3 highest confidence classes from prediction:
            if (i < 3):
                top_3_classes.append(labels[cur_class])
                top_3_classes.append(cur_score)
        
        # Print correspoding name of classes and confidence level:
        print('Predicting ' + k + '...')
        print(top_3_classes)
        # Assign name of image file:
        csv_content[0] = k
        for j in range(len(top_3_classes)):
            csv_content[j+1] = top_3_classes[j]
        # Write into csv file:
        wr.writerow(csv_content)
    print("\nSuccessfully generated all test results in model_classify_result.csv")