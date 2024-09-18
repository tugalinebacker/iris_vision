import torch
from PIL import Image
import cv2
import numpy as np

# Load your YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/tugalinebacker/Desktop/v5nano.pt')  # Replace 'best.pt' with your trained model path

# Load an image
img = cv2.imread('/home/tugalinebacker/Desktop/vidar.jpg')  # Replace 'image.jpg' with your image path

# Run inference
results = model(img)


results.print() 


# Option 1: Display image with bounding boxes using YOLOv5's built-in function
results.show()  # This will open the image in a window with the bounding boxes

# Option 2: Display image with bounding boxes using OpenCV
# Convert the PIL image to a NumPy array
img_np = np.array(img)

# Convert RGB to BGR for OpenCV
img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Draw bounding boxes on the image
for *xyxy, conf, cls in results.xyxy[0]:  # xyxy are the coordinates, conf is the confidence, and cls is the class index
    x1, y1, x2, y2 = map(int, xyxy)  # Convert to integer
    label = f'{model.names[int(cls)]} {conf:.2f}'  # Class name and confidence

    # Draw the bounding box and label on the image
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
    cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with bounding boxes using OpenCV
cv2.imshow('YOLOv5 Detection', img_np)
cv2.waitKey(0)  # Press any key to close the window
cv2.destroyAllWindows()