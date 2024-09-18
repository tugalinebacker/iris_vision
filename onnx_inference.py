from ultralytics import YOLO
import cv2
import numpy as np

onnx_model = YOLO("/home/tugalinebacker/catkin_stonefish/src/iris_vision/src/yolov8_trainings_mkII/best.onnx")

results = onnx_model("/home/tugalinebacker/Desktop/LASMU/stonefish_dataset/test6.png")

# Load the image using OpenCV
image = cv2.imread("/home/tugalinebacker/Desktop/LASMU/stonefish_dataset/test6.png")

# Convert results to numpy array
boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
scores = results[0].boxes.conf.cpu().numpy()  # Get scores
classes = results[0].boxes.cls.cpu().numpy()  # Get class labels

# Define colors for bounding boxes
colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=int)

# Draw bounding boxes on the image
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    color = colors[i].tolist()
    label = f"Class: {int(classes[i])}, Score: {scores[i]:.2f}"
    
    # Draw rectangle
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Put text
    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("Inference Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()