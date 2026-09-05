"""Pose detection helper built on the MediaPipe Tasks vision API.

The original version of this module used ``mediapipe.solutions.pose``. Google
removed that legacy Solutions namespace, so the old code raises
``AttributeError: module 'mediapipe' has no attribute 'solutions'`` on every
mediapipe currently on PyPI. This rewrite uses ``PoseLandmarker`` instead and
keeps the same three public methods, so callers do not change.

The landmark numbering is unchanged (33 point BlazePose topology), so index 13
is still the left elbow and index 25 is still the left knee.
"""

import math
import os
import time
import urllib.request

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Landmark indices worth naming, so the exercise code reads as anatomy.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def ensure_model(model="lite"):
    """Return a local path to the .task model, downloading it once if needed."""
    if model not in MODEL_URLS:
        raise ValueError(f"model must be one of {sorted(MODEL_URLS)}, got {model!r}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"pose_landmarker_{model}.task")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"downloading pose_landmarker_{model} model, one time only")
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URLS[model], tmp)
        os.replace(tmp, path)
    return path


class PoseDetector:
    """Detects a single body per frame and measures joint angles.

    static_image_mode
        False runs the tracking pipeline for video, which is what the webcam
        trainer wants. True treats every call as an unrelated photo.
    min_visibility
        Landmarks below this confidence are treated as not seen at all, which
        stops the angle maths from reading noise when a joint is off frame.
    """

    def __init__(self, static_image_mode=False, model="lite", detectionCon=0.5,
                 trackCon=0.5, presenceCon=0.5, min_visibility=0.5, model_path=None):
        self.static_image_mode = static_image_mode
        self.min_visibility = min_visibility

        # These two are initialised here so calling get_position() or
        # get_angle() before get_pose() returns empty instead of raising
        # AttributeError, which is what the previous version did.
        self.result = None
        self.lmList = []

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path or ensure_model(model)
        )
        running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=detectionCon,
            min_pose_presence_confidence=presenceCon,
            min_tracking_confidence=trackCon,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._connections = PoseLandmarksConnections.POSE_LANDMARKS
        self._t0 = time.monotonic()
        self._last_ts = -1

    def _timestamp_ms(self):
        """Video mode needs timestamps that always move forward."""
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def get_pose(self, img, draw=True):
        """Run detection on one BGR frame and optionally draw the skeleton."""
        if img is None:
            self.result = None
            return img

        # MediaPipe expects RGB. The old code handed it the raw BGR frame,
        # which quietly cost accuracy on every single frame.
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.static_image_mode:
            self.result = self.landmarker.detect(mp_image)
        else:
            self.result = self.landmarker.detect_for_video(
                mp_image, self._timestamp_ms()
            )

        if draw:
            self.draw_landmarks(img)
        return img

    def _landmarks(self):
        if self.result and self.result.pose_landmarks:
            return self.result.pose_landmarks[0]
        return None

    def draw_landmarks(self, img, point_color=(0, 210, 255), line_color=(235, 235, 235)):
        landmarks = self._landmarks()
        if landmarks is None:
            return img
        h, w = img.shape[:2]
        pts = {}
        for idx, lm in enumerate(landmarks):
            if lm.visibility is not None and lm.visibility < self.min_visibility:
                continue
            pts[idx] = (int(lm.x * w), int(lm.y * h))
        for c in self._connections:
            if c.start in pts and c.end in pts:
                cv.line(img, pts[c.start], pts[c.end], line_color, 2, cv.LINE_AA)
        for p in pts.values():
            cv.circle(img, p, 4, point_color, cv.FILLED, cv.LINE_AA)
        return img

    def get_position(self, img, draw=True):
        """Return [[id, x, y], ...] in pixels for the landmarks actually seen."""
        self.lmList = []
        landmarks = self._landmarks()
        if landmarks is None:
            return self.lmList
        h, w = img.shape[:2]
        for idx, lm in enumerate(landmarks):
            if lm.visibility is not None and lm.visibility < self.min_visibility:
                continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.lmList.append([idx, cx, cy])
            if draw:
                cv.circle(img, (cx, cy), 8, (255, 0, 0), cv.FILLED)
        return self.lmList

    def _point(self, idx):
        for lid, x, y in self.lmList:
            if lid == idx:
                return x, y
        return None

    def get_angle(self, img, p1, p2, p3, draw=True):
        """Angle at joint p2 between limbs p2->p1 and p2->p3, in degrees.

        Returns None when any of the three joints was not seen, so callers can
        skip the frame instead of counting a rep from missing data.
        """
        a, b, c = self._point(p1), self._point(p2), self._point(p3)
        if a is None or b is None or c is None:
            return None

        x1, y1 = a
        x2, y2 = b
        x3, y3 = c

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        angle = abs(angle) % 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3, cv.LINE_AA)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3, cv.LINE_AA)
            for p in ((x1, y1), (x2, y2), (x3, y3)):
                cv.circle(img, p, 10, (255, 0, 0), 3, cv.LINE_AA)
            cv.putText(img, str(int(angle)), (x2 - 20, y2 + 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2, cv.LINE_AA)
        return angle

    def close(self):
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
