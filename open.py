import numpy as np
from ultralytics import YOLO 
import cv2 
import cvzone 
import math
from sort import *

# Initialize video capture
cap = cv2.VideoCapture("cot vid.mp4")
if not cap.isOpened():
    raise ValueError("Could not open video file")

# Get video dimensions for proper mask sizing
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = YOLO("yolov8s.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load and prepare mask
mask = cv2.imread("mask.png")
if mask is None:
    # Create default mask if none exists
    mask = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    print("Created default white mask")
else:
    mask = cv2.resize(mask, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Enhanced line definitions for complete stair coverage
limitsUp = [100, 150, 300, 150]       # Upper stair line (adjust coordinates)
limitsDown = [500, 450, 700, 450]     # Lower stair line (adjust coordinates)
line_thickness = 3
line_color = (0, 0, 255)              # Red for inactive lines
active_color = (0, 255, 0)            # Green when crossed

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Apply mask
    imgRegion = cv2.bitwise_and(img, mask)

    # Create dynamic graphics overlay
    overlay = np.zeros((100, 300, 4), dtype=np.uint8)
    
    # Up counter (green)
    cv2.putText(overlay, f"Up: {len(totalCountUp)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)
    cv2.arrowedLine(overlay, (150, 20), (180, 20), (0, 255, 0, 255), 2)
    
    # Down counter (red)
    cv2.putText(overlay, f"Down: {len(totalCountDown)}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255, 255), 2)
    cv2.arrowedLine(overlay, (150, 60), (120, 60), (0, 0, 255, 255), 2)
    
    img = cvzone.overlayPNG(img, overlay, (10, 10))

    # Detection and tracking
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker
    resultsTracker = tracker.update(detections)

    # Draw counting lines
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), line_color, line_thickness)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), line_color, line_thickness)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        # Draw bounding box and ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                         scale=2, thickness=3, offset=10)

        # Calculate center point
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check line crossings with improved logic
        up_crossing = (limitsUp[0] < cx < limitsUp[2]) and (abs(cy - limitsUp[1]) < 20)
        down_crossing = (limitsDown[0] < cx < limitsDown[2]) and (abs(cy - limitsDown[1]) < 20)

        if up_crossing and id not in totalCountUp:
            totalCountUp.append(id)
            cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), active_color, line_thickness+2)

        if down_crossing and id not in totalCountDown:
            totalCountDown.append(id)
            cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), active_color, line_thickness+2)

    # Display large counters
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("People Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
