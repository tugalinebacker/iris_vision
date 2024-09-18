from ultralytics import YOLO
import cv2

# Load your YOLOv8 model
model = YOLO('/home/tugalinebacker/catkin_stonefish/src/iris_vision/src/yolov8_trainings_mkII/models/hms_surprise_v8/yolov8n_coslr_e200_b64/weights/best.pt')  # Replace 'best.pt' with your trained model path

# Load an image
img = cv2.imread('/home/tugalinebacker/Desktop/vidar.jpg')  # Replace 'image.jpg' with your image path

# Run inference
results = model(img)


result = results[0]

# Get bounding boxes, confidences, and class IDs
boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
confidences = result.boxes.conf  # Confidence scores
class_ids = result.boxes.cls  # Class IDs

# Iterate over the detected objects and draw bounding boxes
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integer
    confidence = confidences[i]  # Get confidence for the current box
    class_id = int(class_ids[i])  # Get class ID for the current box

    # Draw the bounding box on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Add class label and confidence on top of the bounding box
    label = f"Class {class_id}: {confidence:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue text

# Show the image with bounding boxes
cv2.imshow('YOLOv8 Detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



