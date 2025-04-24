import cv2
import numpy as np
from gtts import gTTS
import pygame
import time
import os

# --- INIT AUDIO FILES ---
ALERT_FILE = "alert_drowsy.mp3"
PERSON_ALERT_FILE = "alert_person.mp3"
VEHICLE_ALERT_FILE = "alert_vehicle.mp3"

def generate_audio_if_not_exists(filename, text):
    if not os.path.exists(filename):
        tts = gTTS(text, lang='en')
        tts.save(filename)

generate_audio_if_not_exists(ALERT_FILE, "Wake up! You are drowsy!")
generate_audio_if_not_exists(PERSON_ALERT_FILE, "Person is too close to the vehicle!")
generate_audio_if_not_exists(VEHICLE_ALERT_FILE, "Vehicle is too close to you!")

# --- INIT PYGAME AUDIO ---
pygame.mixer.init()
drowsy_sound = pygame.mixer.Sound(ALERT_FILE)
person_sound = pygame.mixer.Sound(PERSON_ALERT_FILE)
vehicle_sound = pygame.mixer.Sound(VEHICLE_ALERT_FILE)

drowsy_channel = pygame.mixer.Channel(1)
person_channel = pygame.mixer.Channel(2)
vehicle_channel = pygame.mixer.Channel(3)

# Alert tracking
alert_start_time = {"drowsy": 0, "person": 0, "vehicle": 0}
alert_duration = 5  # seconds

# --- Load Classifiers ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier("C:/Users/y.yugandhar/Desktop/PROJECT/haar cascade files/haarcascade_mcs_mouth.xml")

# --- Load YOLO ---
net = cv2.dnn.readNet("C:/Users/y.yugandhar/Desktop/PROJECT/yolo/yolov4-tiny.weights", "C:/Users/y.yugandhar/Desktop/PROJECT/yolo/yolov4-tiny.cfg")
with open("C:/Users/y.yugandhar/Desktop/PROJECT/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# --- Video Sources ---
internal_cam = cv2.VideoCapture(0)
external_cam = cv2.VideoCapture("http://192.168.43.1:4747/video")
if not external_cam.isOpened():
    external_cam = cv2.VideoCapture(1)

external_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
external_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Drowsiness tracking
prev_gray = None
prev_eye_points = None
drowsy_counter = 0
alert_threshold = 20

# --- Alert functions ---
def speak_alert(flag_name):
    current_time = time.time()
    if flag_name == "drowsy" and not drowsy_channel.get_busy():
        drowsy_channel.play(drowsy_sound)
        alert_start_time["drowsy"] = current_time
    elif flag_name == "person" and not person_channel.get_busy():
        person_channel.play(person_sound)
        alert_start_time["person"] = current_time
    elif flag_name == "vehicle" and not vehicle_channel.get_busy():
        vehicle_channel.play(vehicle_sound)
        alert_start_time["vehicle"] = current_time

def stop_alert_if_expired(flag_name, channel):
    if channel.get_busy():
        elapsed = time.time() - alert_start_time[flag_name]
        if elapsed >= alert_duration:
            channel.stop()

# Optical Flow params
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret1, frame1 = internal_cam.read()
    if not ret1:
        continue
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    eyes_detected = False
    mouth_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame1[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            eye_center = np.array([[ex + ew // 2, ey + eh // 2]], dtype=np.float32)
            if prev_gray is not None and prev_eye_points is not None:
                new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_eye_points, None, **lk_params)
                drift = np.linalg.norm(new_points - prev_eye_points)
                if drift < 1:
                    drowsy_counter += 1
            prev_eye_points = eye_center

        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.1, 8, minSize=(30, 30))
        for (mx, my, mw, mh) in mouths:
            if mh > h * 0.3:
                mouth_detected = True
                drowsy_counter += 2

    prev_gray = gray.copy()

    if not eyes_detected and not mouth_detected:
        drowsy_counter += 1
    else:
        drowsy_counter = max(0, drowsy_counter - 2)

    if drowsy_counter > alert_threshold:
        cv2.putText(frame1, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        speak_alert("drowsy")
    stop_alert_if_expired("drowsy", drowsy_channel)

    # External camera (YOLO)
    for _ in range(5):
        external_cam.grab()
    ret2, frame2 = external_cam.read()
    if not ret2:
        continue

    height, width, _ = frame2.shape
    blob = cv2.dnn.blobFromImage(frame2, 0.00392, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    person_detected_close = False
    vehicle_detected_close = False

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                label = classes[class_id]
                color = (0, 255, 0) if label == "car" else (0, 0, 255)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label == "person" and w * h > 10000:
                    person_detected_close = True
                if label in ["car", "truck", "bus"] and w * h > 15000:
                    vehicle_detected_close = True

    if person_detected_close:
        speak_alert("person")
    stop_alert_if_expired("person", person_channel)

    if vehicle_detected_close:
        speak_alert("vehicle")
    stop_alert_if_expired("vehicle", vehicle_channel)

    # --- Timer Overlay ---
    for flag, channel in [("drowsy", drowsy_channel), ("person", person_channel), ("vehicle", vehicle_channel)]:
        if channel.get_busy():
            remaining = int(alert_duration - (time.time() - alert_start_time[flag]))
            if flag == "drowsy":
                cv2.putText(frame1, f"{flag.upper()} ALERT: {remaining}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame2, f"{flag.upper()} ALERT: {remaining}s", (10, 30 if flag == "person" else 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show windows
    cv2.imshow("Drowsiness Detection", frame1)
    cv2.imshow("Object Detection", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- RELEASE EVERYTHING ---
internal_cam.release()
external_cam.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

# --- SAFE CLEANUP AUDIO FILES ---
drowsy_channel.stop()
person_channel.stop()
vehicle_channel.stop()

drowsy_sound = None
person_sound = None
vehicle_sound = None

time.sleep(1)  # Let OS release file locks

for file in [ALERT_FILE, PERSON_ALERT_FILE, VEHICLE_ALERT_FILE]:
    try:
        if os.path.exists(file):
            os.remove(file)
    except PermissionError as e:
        print(f"Error deleting {file}: {e}")
