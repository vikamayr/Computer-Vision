import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2

torch.cuda.is_available()

# Test that torch works
x = torch.rand(5, 3)
print(x)

# Download model from pytorch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# realtime detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()