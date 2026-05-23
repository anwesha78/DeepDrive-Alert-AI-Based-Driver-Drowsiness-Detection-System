# ================= drowsiness_metrics.py =================

import cv2
import dlib
import threading
import os
import time
from collections import deque
from imutils import face_utils
from scipy.spatial import distance as dist
from datetime import datetime
import csv

# ================= PARAMETERS =================

EAR_THRESHOLD   = 0.25
CLOSE_SECONDS   = 2.0
REPEAT_INTERVAL = 3.0
SMOOTH_WINDOW   = 8

# ================= ALARM =================
# Simple threading — os.system say works reliably on Mac

def play_alarm(text):
    def _speak():
        os.system(f'say -r 150 "{text}"')
    t = threading.Thread(target=_speak, daemon=True)
    t.start()

# ================= EAR =================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ================= METRICS =================

class MetricsLogger:
    def __init__(self):
        self.session_start    = time.time()
        self.TP               = 0
        self.TN               = 0
        self.FP               = 0
        self.FN               = 0
        self.response_times   = []
        self.false_alarms     = []
        self.events           = []
        self._gt              = "UNKNOWN"
        self._event_start     = None
        self._alarm_in_event  = False
        self._closed_start    = None

    def press_label(self, label, t):
        # score the event that just ended
        self._end_event(t)
        # start new event
        self._gt             = label
        self._event_start    = t
        self._alarm_in_event = False
        if label == "CLOSED":
            self._closed_start = t

    def alarm_fired(self, t):
        if self._gt == "UNKNOWN" or self._alarm_in_event:
            return
        self._alarm_in_event = True
        if self._gt == "CLOSED":
            self.TP += 1
            if self._closed_start is not None:
                self.response_times.append(t - self._closed_start)
        else:
            self.FP += 1
            self.false_alarms.append(t)
        self.events.append({"time_s": round(t - self.session_start, 2),
                            "ground_truth": self._gt,
                            "outcome": "TP" if self._gt == "CLOSED" else "FP"})

    def _end_event(self, t):
        if self._gt == "UNKNOWN" or self._event_start is None:
            return
        if t - self._event_start < 0.5:
            return
        if not self._alarm_in_event:
            if self._gt == "CLOSED":
                self.FN += 1
                self.events.append({"time_s": round(t - self.session_start, 2),
                                    "ground_truth": "CLOSED", "outcome": "FN"})
            elif self._gt == "OPEN":
                self.TN += 1
                self.events.append({"time_s": round(t - self.session_start, 2),
                                    "ground_truth": "OPEN", "outcome": "TN"})

    def finalise(self, t):
        self._end_event(t)

    def print_report(self):
        total = self.TP + self.TN + self.FP + self.FN
        if total == 0:
            print("\n[NO DATA] Press C before closing eyes and O before opening eyes.\n")
            return
        accuracy  = (self.TP + self.TN) / total
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) else 0
        recall    = self.TP / (self.TP + self.FN) if (self.TP + self.FN) else 0
        f1        = (2*precision*recall/(precision+recall)) if (precision+recall) else 0
        dur_min   = (time.time() - self.session_start) / 60
        fa_rate   = len(self.false_alarms) / dur_min if dur_min else 0
        avg_resp  = (sum(self.response_times)/len(self.response_times)
                     if self.response_times else None)
        sep = "=" * 50
        print(f"\n{sep}")
        print("   DROWSINESS DETECTOR - METRICS REPORT")
        print(sep)
        print(f"  Session duration      : {dur_min:.1f} min")
        print(f"  Total events          : {total}")
        print(f"  True  Positives (TP)  : {self.TP}   drowsy caught correctly")
        print(f"  True  Negatives (TN)  : {self.TN}   open eye, silent correctly")
        print(f"  False Positives (FP)  : {self.FP}   false alarms")
        print(f"  False Negatives (FN)  : {self.FN}   missed drowsiness")
        print(sep)
        print(f"  Accuracy              : {accuracy*100:.1f}%")
        print(f"  Precision             : {precision*100:.1f}%")
        print(f"  Recall (Sensitivity)  : {recall*100:.1f}%")
        print(f"  F1 Score              : {f1*100:.1f}%")
        print(f"  False Alarm Rate      : {fa_rate:.2f} per minute")
        if avg_resp is not None:
            print(f"  Avg Response Time     : {avg_resp:.2f}s  (target ~{CLOSE_SECONDS}s)")
        print(sep + "\n")

    def save_csv(self):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"metrics_log_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time_s","ground_truth","outcome"])
            w.writeheader()
            w.writerows(self.events)
        return path

# ================= MAIN =================

def run_metrics():
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lS, lE)  = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rS, rE)  = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ear_buf            = deque(maxlen=SMOOTH_WINDOW)
    smooth_ear         = None
    below_thresh_since = None
    alarm_stage        = 0
    last_alarm_time    = 0.0
    frame_idx          = 0
    ground_truth       = "UNKNOWN"
    logger             = MetricsLogger()

    print("[INFO] Webcam open. Starting detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_idx += 1
        now     = time.time()
        display = cv2.resize(frame, (640, 480))
        H, W    = display.shape[:2]

        # Detection every 2 frames
        if frame_idx % 2 == 0:
            small   = cv2.resize(frame, (320, 240))
            gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces   = detector(gray, 0)
            raw_ear = None
            sx, sy  = W / 320, H / 240

            for face in faces:
                pts     = face_utils.shape_to_np(predictor(gray, face))
                left    = pts[lS:lE]
                right   = pts[rS:rE]
                raw_ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
                cv2.polylines(display, [(left  * [sx, sy]).astype(int)], True, (0,220,0), 1)
                cv2.polylines(display, [(right * [sx, sy]).astype(int)], True, (0,220,0), 1)
                break

            if raw_ear is not None:
                ear_buf.append(raw_ear)
            smooth_ear = sum(ear_buf) / len(ear_buf) if ear_buf else None

        # ===== DROWSINESS LOGIC =====

        alarm_now = False

        if smooth_ear is None:
            status = "NO FACE"
            below_thresh_since = None
            alarm_stage = 0

        elif smooth_ear >= EAR_THRESHOLD:
            status = "EYES OPEN"
            below_thresh_since = None
            alarm_stage = 0

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
                    alarm_stage     = 1
                    last_alarm_time = now
                    alarm_now       = True
                    logger.alarm_fired(now)

                elif now - last_alarm_time >= REPEAT_INTERVAL:
                    play_alarm("Wake up! Wake up!")
                    last_alarm_time = now
                    alarm_now       = True
                    logger.alarm_fired(now)

        # ===== KEY INPUT =====

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('C')):
            ground_truth = "CLOSED"
            logger.press_label("CLOSED", now)
        elif key in (ord('o'), ord('O')):
            ground_truth = "OPEN"
            logger.press_label("OPEN", now)
        elif key in (ord('q'), ord('Q'), 27):
            break

        # ===== DISPLAY =====

        # Status
        col = (0, 255, 0) if status == "EYES OPEN" else (0, 0, 255)
        cv2.putText(display, status, (160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3)

        # EAR
        if smooth_ear is not None:
            ec = (0,255,0) if smooth_ear >= EAR_THRESHOLD else (0,0,255)
            cv2.putText(display, f"EAR: {smooth_ear:.3f}  thresh: {EAR_THRESHOLD}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ec, 1)

        # Ground truth label
        gc = (0,255,255) if ground_truth=="OPEN" else \
             (0,100,255) if ground_truth=="CLOSED" else (180,180,180)
        cv2.putText(display, f"Label: {ground_truth}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gc, 2)

        # Counters
        cv2.putText(display,
                    f"TP:{logger.TP}  TN:{logger.TN}  FP:{logger.FP}  FN:{logger.FN}",
                    (10, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        # Instructions
        cv2.putText(display, "C=close eyes  O=open eyes  Q=quit+report",
                    (100, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

        # Red flash when alarm fires
        if alarm_now:
            overlay = display.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (0,0,180), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

        cv2.imshow("Drowsiness Metrics Mode", display)

    # ===== SHUTDOWN =====
    logger.finalise(time.time())
    cap.release()
    cv2.destroyAllWindows()
    csv_path = logger.save_csv()
    logger.print_report()
    print(f"[INFO] CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    print("[INFO] drowsiness_metrics.py")
    print("  Press C  ->  then immediately close eyes")
    print("  Press O  ->  then immediately open eyes")
    print("  Press Q  ->  quit and see full metrics report\n")
    run_metrics()
