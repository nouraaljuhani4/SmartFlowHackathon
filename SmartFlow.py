from ultralytics import YOLO
import cv2
import numpy as np
import json
from datetime import datetime
import winsound

# -------- CONFIG --------
USE_VIDEO = True
VIDEO_PATH = r"C:\Users\aljuh\OneDrive\Documents\APRIL 2025\Python-3.11.14\traffic2.mp4.mp4"

CAMERA_ID = "CAM_01"
CAMERA_NAME = "Hackathon Camera 1"
STATUS_FILE = "status.json"
ALERT_LOG_FILE = "alerts.log"

# Example coordinates (change if you want)
CAMERA_LAT = 24.687060
CAMERA_LON = 46.708943

# Incident detection
STOPPED_FRAME_THRESHOLD = 240     # frames "not moving" to consider incident
MOVEMENT_TOLERANCE = 5            # pixels movement considered "still"
MAX_ASSOCIATION_DISTANCE = 50     # pixels to match detections to existing IDs


def get_congestion_level(count):
    if count < 10:
        return "Low Traffic", (0, 255, 0)
    elif count < 17:
        return "Medium Traffic", (0, 255, 255)
    else:
        return "High Traffic", (0, 0, 255)


def play_alert_sound():
    # Beep sound on Windows
    winsound.Beep(1000, 400)


def log_alert(reason, level, vehicle_count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {reason} | Level: {level} | Vehicles: {vehicle_count}\n"
    with open(ALERT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def update_status_json(vehicle_count, level, incident_count):
    status = {
        CAMERA_ID: {
            "name": CAMERA_NAME,
            "lat": CAMERA_LAT,
            "lon": CAMERA_LON,
            "vehicles": vehicle_count,
            "level": level,
            "incidents": incident_count,
            "timestamp": datetime.now().isoformat()
        }
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=4)


class SimpleTracker:
    """
    Very simple centroid tracker (no lap, no C++ libs).
    Gives each vehicle an ID and tracks it across frames using distance.
    Also keeps 'stopped' frame counts for incident detection.
    """

    def __init__(self, max_distance=50, movement_tol=5, stopped_frames_thresh=60):
        self.next_id = 1
        self.objects = {}  # id -> {"centroid": (x,y), "stopped_frames": int}
        self.max_distance = max_distance
        self.movement_tol = movement_tol
        self.stopped_frames_thresh = stopped_frames_thresh

    def update(self, detections):
        """
        detections: list of (cx, cy)
        Returns: list of (id, cx, cy, is_stopped)
        """
        results = []

        if len(self.objects) == 0:
            # No existing objects: assign new IDs to all detections
            for (cx, cy) in detections:
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = {"centroid": (cx, cy), "stopped_frames": 0}
                results.append((obj_id, cx, cy, False))
            return results

        # For each detection, find closest existing object within max_distance
        used_ids = set()
        for (cx, cy) in detections:
            best_id = None
            best_dist = None
            for obj_id, data in self.objects.items():
                if obj_id in used_ids:
                    continue
                px, py = data["centroid"]
                dist = np.hypot(cx - px, cy - py)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = obj_id

            if best_id is not None and best_dist is not None and best_dist <= self.max_distance:
                # Update existing object
                px, py = self.objects[best_id]["centroid"]
                dx = abs(cx - px)
                dy = abs(cy - py)
                if dx < self.movement_tol and dy < self.movement_tol:
                    self.objects[best_id]["stopped_frames"] += 1
                else:
                    self.objects[best_id]["stopped_frames"] = 0
                self.objects[best_id]["centroid"] = (cx, cy)
                used_ids.add(best_id)
                is_stopped = self.objects[best_id]["stopped_frames"] >= self.stopped_frames_thresh
                results.append((best_id, cx, cy, is_stopped))
            else:
                # Create new object
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = {"centroid": (cx, cy), "stopped_frames": 0}
                used_ids.add(obj_id)
                results.append((obj_id, cx, cy, False))

        # (We ignore objects not matched in this frame; for a simple hackathon tracker this is fine.)
        return results


def main():
    model = YOLO("yolov8n.pt")

    if USE_VIDEO:
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Cannot open source (video or camera).")
        return

    tracker = SimpleTracker(
        max_distance=MAX_ASSOCIATION_DISTANCE,
        movement_tol=MOVEMENT_TOLERANCE,
        stopped_frames_thresh=STOPPED_FRAME_THRESHOLD,
    )

    print(" SmartFlow running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot read from source.")
            break

        # ----- YOLO detection (NO model.track → NO lap needed) -----
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        vehicle_count = 0
        incident_count = 0
        detections = []  # list of (cx, cy) for tracking
        box_data = []    # store xyxy for drawing along with IDs

        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), cls in zip(xyxy, classes):
                cls = int(cls)
                # COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
                if cls in [2, 3, 5, 7]:
                    vehicle_count += 1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    detections.append((cx, cy))
                    box_data.append((x1, y1, x2, y2))

        # ----- Simple tracking to assign IDs and detect stopped vehicles -----
        tracked_info = tracker.update(detections)  # list of (id, cx, cy, is_stopped)

        # Draw boxes + IDs
        for ((x1, y1, x2, y2), (obj_id, cx, cy, is_stopped)) in zip(box_data, tracked_info):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # ID label
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Incident marking
            if is_stopped:
                incident_count += 1
                cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)

        # ----- Congestion level -----
        level, color = get_congestion_level(vehicle_count)

        # ----- Alerts (sound + log) -----
        if level == "High Traffic" or incident_count > 0:
            play_alert_sound()
            reason = "High congestion" if level == "High Traffic" else "Incident detected"
            log_alert(reason, level, vehicle_count)

        # ----- Write JSON for heatmap/dashboard -----
        update_status_json(vehicle_count, level, incident_count)

        # ----- Overlay (HUD) -----
        cv2.putText(frame, f"{CAMERA_NAME}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Congestion: {level}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Incidents: {incident_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("SmartFlow - Traffic Monitoring (IDs + Incidents)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ SmartFlow stopped.")


if __name__ == "__main__":
    main()
