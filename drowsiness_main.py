# ================= FINAL CLEAN DEMO VERSION =================

import cv2
import dlib
import threading
import subprocess
import time
from collections import deque
from imutils import face_utils
from scipy.spatial import distance as dist

# ================= PARAMETERS =================

EAR_THRESHOLD   = 0.23
CLOSE_SECONDS   = 2.0
REPEAT_INTERVAL = 2.5
SMOOTH_WINDOW   = 8

# ================= ALARM =================

_alarm_event = threading.Event()
_alarm_text_store = [""]
_alarm_running = False

def stop_voice():
    try:
        subprocess.run(["killall", "say"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except:
        pass

def _alarm_worker():
    global _alarm_running
    while True:
        _alarm_event.wait()
        _alarm_event.clear()
        _alarm_running = True
        try:
            subprocess.Popen(["say", "-r", "160", _alarm_text_store[0]])
        except:
            pass
        time.sleep(0.1)
        _alarm_running = False

threading.Thread(target=_alarm_worker, daemon=True).start()

def play_alarm(text):
    if not _alarm_running:
        _alarm_text_store[0] = text
        _alarm_event.set()

# ================= EAR =================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ================= INIT =================

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lS, lE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rS, rE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

ear_buf = deque(maxlen=SMOOTH_WINDOW)
smooth_ear = None

below_thresh_since = None
alarm_stage = 0
last_alarm_time = 0

frame_idx = 0

# ================= LOOP =================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_idx += 1
    now = time.time()
    display = cv2.resize(frame,(640,480))
    H,W = display.shape[:2]

    # ===== STABLE DETECTION =====
    if frame_idx % 3 == 0:
        small = cv2.resize(frame,(320,240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        sx, sy = W/320, H/240
        raw_ear = None

        for face in faces:
            pts = face_utils.shape_to_np(predictor(gray,face))
            left = pts[lS:lE]
            right = pts[rS:rE]

            raw_ear = (eye_aspect_ratio(left)+eye_aspect_ratio(right))/2

            cv2.polylines(display,[(left*[sx,sy]).astype(int)],True,(0,255,0),1)
            cv2.polylines(display,[(right*[sx,sy]).astype(int)],True,(0,255,0),1)
            break

        # ✅ safe update
        if raw_ear is not None:
            ear_buf.append(raw_ear)

        if len(ear_buf) > 0:
            smooth_ear = sum(ear_buf)/len(ear_buf)

        # ✅ FIXED (no crash now)
        if smooth_ear is not None and raw_ear is not None:
            smooth_ear = 0.7*smooth_ear + 0.3*raw_ear

    # ===== LOGIC =====

    if smooth_ear is None:
        status = "NO FACE"

    elif smooth_ear >= EAR_THRESHOLD:
        stop_voice()
        below_thresh_since = None
        alarm_stage = 0
        status = "EYES OPEN"

    else:
        if below_thresh_since is None:
            below_thresh_since = now

        duration = now - below_thresh_since

        if duration < CLOSE_SECONDS:
            status = "EYES OPEN"

        else:
            status = "DROWSY!"

            if alarm_stage == 0:
                play_alarm("Wake up!")
                alarm_stage = 1
                last_alarm_time = now

            elif now - last_alarm_time > REPEAT_INTERVAL:
                play_alarm("Wake up! Wake up!")
                last_alarm_time = now

    # ===== DISPLAY =====

    color = (0,255,0) if status=="EYES OPEN" else (0,0,255)

    cv2.putText(display, status, (180,80),
                cv2.FONT_HERSHEY_SIMPLEX,1.5,color,3)

    if smooth_ear:
        cv2.putText(display,f"EAR: {smooth_ear:.3f}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    # 🔥 TIMER + PROGRESS BAR
    if below_thresh_since:
        elapsed = now - below_thresh_since
        progress = min(elapsed/CLOSE_SECONDS,1.0)

        cv2.rectangle(display,(20,H-20),(W-20,H-10),(60,60,60),-1)
        cv2.rectangle(display,(20,H-20),
                      (20+int((W-40)*progress),H-10),
                      (0,0,255),-1)

        cv2.putText(display,f"Closed: {elapsed:.1f}s / {CLOSE_SECONDS}s",
                    (20,H-30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow("Driver Drowsiness Detector",display)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()