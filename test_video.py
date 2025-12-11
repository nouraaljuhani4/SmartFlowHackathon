import os
import cv2

video_path = r"C:\Users\aljuh\OneDrive\Documents\APRIL 2025\Python-3.11.14\traffic.mp4.mp4 " # ✅ fixed name and quotes

print("Absolute path:", os.path.abspath(video_path))
print("File exists:", os.path.exists(video_path))

cap = cv2.VideoCapture(video_path)
print("cap.isOpened():", cap.isOpened())

if not cap.isOpened():
    print("❌ OpenCV could NOT open this video. Try another MP4 or convert it.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame (end of file or unsupported format).")
        break

    cv2.imshow("Video test", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done playing video.")
