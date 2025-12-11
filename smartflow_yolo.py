from ultralytics import YOLO
import cv2
import json
import time

# ---- SETTINGS ----
CAMERA_ID = "CAM_01"
CAMERA_NAME = "Location 1"
STATUS_FILE = "status.json"

# ---- LOAD YOLO MODEL ----
model = YOLO("yolov8n.pt")  # make sure yolov8n.pt is in your project folder

# ---- OPEN VIDEO OR CAMERA ----
# 0 = webcam, or use "../videos/traffic.mp4" for file
cap = cv2.VideoCapture("../videos/traffic.mp4")


def congestion_level(count):
    if count < 10:
        return "Low"
    elif count < 25:
        return "Medium"
    else:
        return "High"


def write_status(count, level):
    status = {
        CAMERA_ID: {
            "name": CAMERA_NAME,
            "vehicles": count,
            "level": level,
            "timestamp": time.time()
        }
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=4)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = results[0].boxes.data

    vehicle_count = 0

    for det in detections:
        x1, y1, x2, y2, score, cls = det
        cls = int(cls)

        # car=2, motorcycle=3, bus=5, truck=7
        if cls in [2, 3, 5, 7]:
            vehicle_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)

    level = congestion_level(vehicle_count)

    # write to status.json (in detection folder)
    write_status(vehicle_count, level)

    cv2.putText(frame, f"{CAMERA_NAME}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Level: {level}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("SmartFlow - YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
