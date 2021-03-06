IRIS AUV vision control for Stonefish

Still under development. Ghost net detecion runs on TensorFlow 2.5.0 and ROS Noetic (Ubuntu 20.04), based on various retrained models, such as efficientdet_d0, ssd_resnet50_v1 and ssd_mobilenet_v2 (see exported-models directory). Directory annotations contains a label_map, test and train records obtained from over a 1000 images.
Tests were successfully carried out with CPU (Intel Core i7-8565U) and GPU (NVIDIA GeForce MX250), both with RAM 8 GB. 
It's advisable to install and run TensorFlow in a virtual environment.

For TensorFlow installation: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation

For ROS Noetic instalaltion: http://wiki.ros.org/noetic/Installation

For IRIS_stonefish: https://github.com/lmbalves/iris_stonefish

For TensorFlow 2 Detection Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

For virtual environment (venv): https://docs.python.org/3/library/venv.html

