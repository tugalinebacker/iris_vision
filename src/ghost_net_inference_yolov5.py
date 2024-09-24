#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String

class YOLOv5ROS:
    def __init__(self):
        # Initialize YOLOv5 model - update model path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/tugalinebacker/catkin_stonefish/src/iris_vision/src/exported-models/yolov5n_e300_b128_AdamW_lr0.pt')
        # Initialize a CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()
        # Subscribe to IRIS' front camera
        rospy.Subscriber("/iris/proscilica_front/image_color", Image, self.image_callback)
         #Set publisher with bounding box data
        self.detection_data = rospy.Publisher("/iris/proscilica_front/ghost_detection_data", Float32MultiArray, queue_size=10)

    def image_callback(self, data):
        try:
            # Convert the ROS Image message to a format OpenCV understands (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # COORDINATES OF THE CENTER POINTS OF THE FRAME
            image_height, image_width, _ = cv_image.shape
            x_frame_center_point = image_width/2
            y_frame_center_point = image_height/2

            # Run YOLOv5 inference on the image
            results = self.model(cv_image)
            if results.xyxy[0].shape[0] > 0:
                # Extract bounding boxes, confidence scores, and class indices
                bboxes = results.xyxy[0][:, :4].int().cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                confidences = results.xyxy[0][:, 4].cpu().numpy() # Confidence scores
                class_ids = results.xyxy[0][:, 5].int().cpu().numpy()  # Class indices

                # Get the index of the highest confidence score using np.argmax
                best_idx = np.argmax(confidences)

                x1, y1, x2, y2 = bboxes[best_idx]
                best_conf = confidences[best_idx]
                best_class_id = class_ids[best_idx]
                
                # COORDINATES OF THE HIGHEST SCORING BOUNDING BOX
                x_bbox_center_point = (x1 + x2) // 2
                y_bbox_center_point = (y1 + y2) // 2

                # SIZE OF THE BOUNDING BOX -> width in X, height in Y
                bbox_width = x2-x1
                bbox_height = y2-y1

                #DRAW BOUNDING BOX
                label = f'{self.model.names[best_class_id]} {best_conf:.2f}'
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # STORE COORDINATES and BBOX SIZE IN ROS MESSAGE
                center_points_coordinates = Float32MultiArray()
                # self.captain_data = Bool()
                center_points_coordinates.data = [x_frame_center_point, y_frame_center_point, x_bbox_center_point, y_bbox_center_point, 
                image_width, image_height, bbox_width, bbox_height]

            # Optionally display the image (can comment out for headless setups)
            cv2.imshow("Inference with YOLOv5", cv_image)
            cv2.waitKey(3)

            self.detection_data.publish(center_points_coordinates)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    rospy.init_node('ghost_net_inference_yolov5', anonymous=True)
    yolov5_ros = YOLOv5ROS()
    rospy.spin()
