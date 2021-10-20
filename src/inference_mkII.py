#!/usr/bin/env python3
#coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import rospy
import tensorflow as tf
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from pathlib import Path

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


dirname = os.path.dirname(__file__)
PATH_TO_MODEL_DIR = os.path.join(dirname, 'exported-models/ghost_mkII_efficientdet_d0_GPU/')
# PATH_TO_MODEL_DIR = os.path.join(dirname, 'exported-models/ghost_mkII_resnet50_CPU/')
# PATH_TO_MODEL_DIR = os.path.join(dirname, 'exported-models/ghost_mkII_mobilenet_CPU/')
PATH_TO_LABELS = os.path.join(dirname, 'annotations/label_map.pbtxt')
MIN_CONF_THRESH = 0.6
PATH_TO_SAVED_MODEL = os.path.join(dirname, 'exported-models/ghost_mkII_efficientdet_d0_GPU/saved_model')

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




def callback(img):
    #rospy.loginfo("Topic /iris/proscilica_front/image_color detected!")
    bridge = CvBridge()
    ghost_detection = rospy.Publisher("/iris/ghost_detection", Image)

    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8") #converts ROS image to cv2
    except CvBridgeError as e:
        print(e)
    
    cv_image = cv2.resize(cv_image, (640,640))
    #(rows,cols,channels) = cv_image.shape

    input_tensor = tf.convert_to_tensor(cv_image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = cv_image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_CONF_THRESH,
            agnostic_mode=False)


    cv2.imshow("Inference in front camera", image_with_detections)
    cv2.waitKey(3)

    try:
        ghost_detection.publish(bridge.cv2_to_imgmsg(image_with_detections, "bgr8")) #converts cv2 image to ROS
    except CvBridgeError as e:
        print(e)
        


def inference():
    rospy.init_node('inference_mkII', anonymous=True)
    rospy.Subscriber("/iris/proscilica_front/image_color", Image, callback)
    try:
        2
        rospy.spin()
    except KeyboardInterrupt:
        print("Inference node shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    inference()
