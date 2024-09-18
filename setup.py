from distutils.core import setup
 
setup(
    version='0.0.0',
    scripts=[
        'src/ghost_net_inference.py', 
        'src/ghost_net_inference_yolo.py', 
        'src/ghost_net_inference_yolov5.py', 
        'src/test_node.py', 
        'src/sensor_nav.py'
    ],
    packages=[
        'iris_vision',
        'object_detection',
        'numpy=1.20.3'
    ],
    install_requires=[
        'ultralytics',
        'opencv-python'
    ],
    package_dir={'': 'src'}
)
