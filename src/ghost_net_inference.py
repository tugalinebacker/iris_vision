#!/usr/bin/env python3
#coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import rospy
import tensorflow as tf
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray, String
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

efficientdet_d0 = 'exported-models/ghost_mkII_efficientdet_d0_GPU/'
resnet50_CPU = 'exported-models/ghost_mkII_resnet50_CPU/'
mobilenet_CPU = 'exported-models/ghost_mkII_mobilenet_CPU/'

dirname = os.path.dirname(__file__)
PATH_TO_MODEL_DIR = os.path.join(dirname, efficientdet_d0)
PATH_TO_LABELS = os.path.join(dirname, 'annotations/label_map.pbtxt')
MIN_CONF_THRESH = 0.6
PATH_TO_SAVED_MODEL = os.path.join(dirname, efficientdet_d0 + 'saved_model')

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
    bridge = CvBridge()
    ghost_detection = rospy.Publisher("/iris/proscilica_front/ghost_detection", Image)
    center_points = rospy.Publisher("/iris/proscilica_front/image_center_points", Float32MultiArray)

    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8") #converts ROS image to cv2
    except CvBridgeError as e:
        print(e)
    
    cv_image = cv2.resize(cv_image, (640,640))

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

    #FRAME DIMENSIONS, BOUNDING BOXES COORDINATES, SCORES AND COORDINATES IN PIXELS
    image_height, image_width, _ = image_with_detections.shape
    bounding_boxes = detections['detection_boxes']
    confidence_scores = detections['detection_scores']
    bounding_boxes_pixel = bounding_boxes * np.array([image_height, image_width, image_height, image_width])
    
    # DETERMINE THE INDEX OF THE BOUNDING BOX WITH THE HIGHEST SCORE
    max_confidence_index = np.argmax(confidence_scores)
    
    # VALUE OF THE BOUNDING BOX WITH THE HIGHEST SCORE
    highest_confidence_bbox_score = confidence_scores[max_confidence_index]
    print("\nHighest confidence box score -> " + str(round(highest_confidence_bbox_score,2)))
    
    # COORDINATES OF THE BOUNDING BOX WITH THE HIGHEST SCORE
    highest_confidence_bbox_coordinates = bounding_boxes[max_confidence_index]
    highest_confidence_bbox_coordinates_pixel = bounding_boxes_pixel[max_confidence_index]

    # COORDINATES OF THE CENTER POINTS OF THE FRAME AND HIGHEST SCORING BOUNDING BOX
    x_frame_center_point = image_width/2
    y_frame_center_point = image_height/2
    x_bbox_center_point = highest_confidence_bbox_coordinates_pixel[1]+highest_confidence_bbox_coordinates_pixel[3]/2
    y_bbox_center_point = highest_confidence_bbox_coordinates_pixel[0]+highest_confidence_bbox_coordinates_pixel[2]/2
    print("Box center point coordinates -> ( " + str(round(x_bbox_center_point,2)) + ", " + str(round(y_bbox_center_point,2)) + ")")
    
    # STORE COORDINATES IN ROS MESSAGE
    center_points_coordinates = Float32MultiArray()
    center_points_coordinates.data = [x_frame_center_point, y_frame_center_point, x_bbox_center_point, y_bbox_center_point]

    cv2.imshow("Inference in front camera", image_with_detections)
    cv2.waitKey(3)

    try:
        ghost_detection.publish(bridge.cv2_to_imgmsg(image_with_detections, "bgr8")) #converts cv2 image to ROS
        center_points.publish(center_points_coordinates)
    except CvBridgeError as e:
        print(e)
        


def inference():
    rospy.init_node('ghost_net_inference', anonymous=True)
    rospy.Subscriber("/iris/proscilica_front/image_color", Image, callback)
    try:
        2
        rospy.spin()
    except KeyboardInterrupt:
        print("Inference node shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    inference()
