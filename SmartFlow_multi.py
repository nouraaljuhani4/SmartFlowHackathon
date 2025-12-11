from ultralytics import YOLO
import cv2
import numpy as np
import json
from datetime import datetime
import winsound

# -------- CONFIG --------

# 2 video sources + 1 live camera (index 0)
SOURCES = [
    r"C:\Users\aljuh\OneDrive\Documents\APRIL 2025\Python-3.11.14\traffic2.mp4.mp4",  # CAM_01
    r"C:\Users\aljuh\OneDrive\Documents\APRIL 2025\Python-3.11.14\traffic3.mp4.mp4",  # CAM_02
    0,  # CAM_03 = live webcam
]

# Mark which sources are videos (so we can loop them) and which is a live camera
IS_VIDEO = [True, True, False]

# 3 cameras metadata (IDs, names, coordinates)
CAMERAS = [
    {
        "id": "CAM_01",
        "name": "Highway Camera",
        "lat": 24.687060,
        "lon": 46.708943,
    },
    {
        "id": "CAM_02",
        "name": "City Center Camera",
        "lat": 24.668566,
        "lon": 46.695156,
    },
    {
        "id": "CAM_03",
        "name": "Live Field Camera",
        "lat": 24.665760,
        "lon": 46.682430,
    },
]

STATUS_FILE = "status.json"
ALERT_LOG_FILE = "alerts_multi.log"

# Incident detection parameters
STOPPED_FRAME_THRESHOLD = 240      # frames "not moving" to consider incident
MOVEMENT_TOLERANCE = 5            # pixels movement considered "still"
MAX_ASSOCIATION_DISTANCE = 50     # pixels to match detections to existing IDs


def get_congestion_level(count):
    if count < 10:
        return "Low Traffic", (0, 255, 0)
    elif count < 15:
        return "Medium Traffic", (0, 255, 255)
    else:
        return "High Traffic", (0, 0, 255)


def play_alert_sound():
    # Beep sound on Windows
    winsound.Beep(1000, 300)


def log_alert(camera_name, reason, level, vehicle_count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {camera_name} | {reason} | Level: {level} | Vehicles: {vehicle_count}\n"
    with open(ALERT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


class SimpleTracker:

    def __init__(self, max_distance=50, movement_tol=5, stopped_frames_thresh=60):
        self.next_id = 1
        self.objects = {}
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

        return results


def process_camera_frame(model, frame, tracker, camera_meta):

    camera_name = camera_meta["name"]

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    vehicle_count = 0
    incident_count = 0
    detections = []
    box_data = []

    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), cls in zip(xyxy, classes):
            cls = int(cls)
            # vehicle classes: car=2, motorcycle=3, bus=5, truck=7
            if cls in [2, 3, 5, 7]:
                vehicle_count += 1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append((cx, cy))
                box_data.append((x1, y1, x2, y2))

    # Tracking
    tracked_info = tracker.update(detections)

    # Draw detections and IDs
    for ((x1, y1, x2, y2), (obj_id, cx, cy, is_stopped)) in zip(box_data, tracked_info):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if is_stopped:
            incident_count += 1
            cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    level, color = get_congestion_level(vehicle_count)


    if level == "High Traffic" or incident_count > 0:
        play_alert_sound()
        reason = "High congestion" if level == "High Traffic" else "Incident detected"
        log_alert(camera_name, reason, level, vehicle_count)


    cv2.putText(frame, camera_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Veh: {vehicle_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Cong: {level}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Inc: {incident_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame, vehicle_count, incident_count, level


def main():
    model = YOLO("yolov8n.pt")

    # One tracker per camera
    trackers = [
        SimpleTracker(
            max_distance=MAX_ASSOCIATION_DISTANCE,
            movement_tol=MOVEMENT_TOLERANCE,
            stopped_frames_thresh=STOPPED_FRAME_THRESHOLD,
        )
        for _ in SOURCES
    ]


    caps = []
    for src in SOURCES:
        cap = cv2.VideoCapture(src)
        caps.append(cap)

    # Check opened
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"❌ Cannot open source for camera {CAMERAS[i]['name']}: {SOURCES[i]}")

    print("✅ SmartFlow MULTI (2 videos + 1 live cam) running. Press 'q' to quit.")

    while True:
        frames = []
        status_data = {}

        for i, cap in enumerate(caps):
            camera_meta = CAMERAS[i]
            is_video = IS_VIDEO[i]

            ret, frame = cap.read()

            if not ret:
                if is_video:

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                else:
                    # Live camera failed: use blank frame
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)


            frame, veh_count, inc_count, level = process_camera_frame(
                model, frame, trackers[i], camera_meta
            )

            # Save status for .Json
            status_data[camera_meta["id"]] = {
                "name": camera_meta["name"],
                "lat": camera_meta["lat"],
                "lon": camera_meta["lon"],
                "vehicles": veh_count,
                "level": level,
                "incidents": inc_count,
                "timestamp": datetime.now().isoformat()
            }


            frame_resized = cv2.resize(frame, (426, 240))  # 16:9 small
            frames.append(frame_resized)

        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=4)

        combined = cv2.hconcat(frames)  # [CAM1 | CAM2 | CAM3]

        cv2.imshow("SmartFlow MULTI - 2 Videos + Live Camera", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print("✅ SmartFlow MULTI stopped.")


if __name__ == "__main__":
    main()
