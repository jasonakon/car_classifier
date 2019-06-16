import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
from PIL import ImageFont, ImageDraw

##############################
## Declare requirement inputs:
##############################
#Path to your test image:
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#Path to your model:
MODEL_NAME = os.path.join("trained_models","grab_model")
CWD_PATH = os.getcwd()
#Path to your inference graph:
PATH_TO_INFERENCE_MODEL = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
#Path to your label_map:
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labels.txt')

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

# Create the ractangular outline shape for classifier output:
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline = (244, 66, 66), fill = color)

################################################
## Start classification and image output viewer:
################################################
# Declare image index for user to iterate the test image sample:
image_index = 0
# Declare the min threshold % for classification:
min_thres_classify = 0.5

while(True):
    # Read images into array using cv:
    frame = cv2.imread(TEST_IMAGE_PATHS[image_index])
    # Get shapes from images:
    frame_h, frame_w, channels = frame.shape
    # Extend the dimension for prediction:
    frame_expanded = np.expand_dims(frame, axis=0)

    # Read image using PIL:
    img = Image.open(TEST_IMAGE_PATHS[image_index])
    try:
        # Create drawable from input images:
        draw = ImageDraw.Draw(img, 'RGBA')
    except:
        # Compatibility issue with some unsupport format images:
        print("This image is not supported at this mode, please use classify_image_csv.py to see the output, this can be fixed in time. Thank you")
        image_index += 1
        # Make sure the program never quit randomly
        continue
    # Declare font use for labeling in image:
    font_calibri = ImageFont.truetype("./utils/fonts/Calibri.ttf", size=20)

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

          print('Image ' + str(image_index) + ' : ' + str(labels[cur_class]), 'score = ', cur_score)

          box = [x_min, y_min, x_max, y_max]
          # Draw boxes in the image:
          draw_rectangle(draw, box, (0,128,128,20), width=5)
          # Add class name and confidence level into the image:
          draw.text((10, 10), 'Press D - Next Photo / Press A - Previous Photo', fill=(255,255,255,20))
          draw.text((box[0] + 20, box[1] - 50), labels[cur_class], fill=(255,255,255,20), font=font_calibri)
          draw.text((box[0] + 20, box[1] - 25), str(cur_score), fill=(255,255,255,20), font=font_calibri)

    # Convert PIL image type to CV2 for display:
    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    # Images compiled with draw overlay and ready for display:
    cv2.imshow('Grab Car Classifer', open_cv_image)

    while True :
        if cv2.waitKey(1) == ord('a'):
            image_index-=1
            break
        if cv2.waitKey(1) == ord('d'):
            image_index+=1
            break
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

