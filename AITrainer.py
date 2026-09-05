"""Rep counter and form feedback for bicep curls and squats.

Run it with no arguments and it opens the workout chooser, then your webcam:

    python main.py

Or skip the dialog and point it at a file, which also works on a machine with
no display:

    python main.py --exercise curl --source clip.mp4 --out annotated.mp4
"""

import argparse
import sys
import time

import cv2 as cv
import numpy as np

import PoseModule as pm

# Joint triples per exercise: the angle is measured at the middle joint.
EXERCISES = {
    "curl": {
        "label": "Bicep Curl",
        # elbow angle, both arms, so the count follows whichever arm is seen
        "joints": [(pm.L_SHOULDER, pm.L_ELBOW, pm.L_WRIST),
                   (pm.R_SHOULDER, pm.R_ELBOW, pm.R_WRIST)],
        # angle at the top of the rep, angle at the bottom
        "range": (50, 160),
    },
    "squat": {
        "label": "Squat",
        # knee angle, the standard depth measure for a squat
        "joints": [(pm.L_HIP, pm.L_KNEE, pm.L_ANKLE),
                   (pm.R_HIP, pm.R_KNEE, pm.R_ANKLE)],
        "range": (70, 165),
    },
}


class RepCounter:
    """Counts a rep as two halves: reaching the top, then returning.

    Kept apart from the video loop so the counting rule can be tested without
    a camera.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.count = 0.0
        self.direction = 0
        self.percent = 0.0

    def update(self, angle):
        """Feed one joint angle. Returns the completed rep count so far."""
        if angle is None:
            return self.count
        self.percent = float(np.interp(angle, (self.low, self.high), (100, 0)))
        if self.percent >= 99 and self.direction == 0:
            self.count += 0.5
            self.direction = 1
        elif self.percent <= 1 and self.direction == 1:
            self.count += 0.5
            self.direction = 0
        return self.count

    @property
    def reps(self):
        return int(self.count)


def choose_exercise():
    """Small Tk dialog. Returns None if the window is closed without a pick."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter is not installed, pass --exercise instead")
        return None

    chosen = {"value": None}

    def pick(name):
        chosen["value"] = name
        win.destroy()

    win = tk.Tk()
    win.title("AI Trainer")
    ttk.Label(win, text="Choose your workout", padding=10).pack()
    for key, cfg in EXERCISES.items():
        ttk.Button(win, text=cfg["label"],
                   command=lambda k=key: pick(k)).pack(fill="x", padx=10, pady=4)
    win.mainloop()
    # Closing the window with the X leaves this as None, which the caller
    # handles. The previous version carried on and crashed later with
    # NameError on an unset percentage.
    return chosen["value"]


def open_source(source):
    """Accept a camera index or a file path and fail with a clear message."""
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        print(f"could not open video source {source!r}")
        return None
    return cap


def measure(detector, img, exercise, draw=True):
    """Return the joint angle for whichever side is fully visible."""
    for triple in EXERCISES[exercise]["joints"]:
        angle = detector.get_angle(img, *triple, draw=draw)
        if angle is not None:
            return angle
    return None


def draw_hud(img, count, percent, angle, exercise, fps):
    h, w = img.shape[:2]
    # Geometry follows the frame size. The old version hardcoded a bar from
    # y=300 to y=600, which fell off the bottom of any 480p webcam.
    top, bottom = int(h * 0.25), int(h * 0.85)
    left, right = int(w * 0.04), int(w * 0.04) + max(24, int(w * 0.045))
    filled = int(np.interp(percent, (0, 100), (bottom, top)))

    at_end = percent >= 99 or percent <= 1
    color = (0, 0, 255) if at_end else (255, 255, 0)

    cv.rectangle(img, (left, top), (right, bottom), (255, 255, 255), cv.FILLED)
    cv.rectangle(img, (left, filled), (right, bottom), color, cv.FILLED)
    cv.rectangle(img, (left, top), (right, bottom), (60, 60, 60), 2)
    cv.putText(img, f"{int(percent)}%", (left, top - 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)

    scale = max(0.9, h / 500)
    cv.putText(img, str(int(count)), (left, int(h * 0.18)),
               cv.FONT_HERSHEY_SIMPLEX, scale * 1.6, (0, 0, 0), int(scale * 4), cv.LINE_AA)

    label = EXERCISES[exercise]["label"]
    footer = f"{label}   angle {int(angle)}" if angle is not None else f"{label}   no body detected"
    cv.putText(img, footer, (left, h - 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, f"{fps:.0f} fps", (w - 110, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
    return img


def countdown(cap, seconds, show):
    """Give the user time to get in frame. Skipped when there is no window."""
    if not show or seconds <= 0:
        return True
    remaining = seconds
    last = time.time()
    while remaining > 0:
        ok, img = cap.read()
        if not ok or img is None:
            # The old version drew on the frame without checking, so a dropped
            # frame here crashed inside cv.putText.
            print("video source stopped during the countdown")
            return False
        h, w = img.shape[:2]
        cv.putText(img, str(remaining), (w // 2 - 30, h // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5, cv.LINE_AA)
        cv.imshow("AI Trainer", img)
        if time.time() - last >= 1:
            last = time.time()
            remaining -= 1
        if cv.waitKey(1) == 27:
            return False
    return True


def run(exercise, source=0, show=True, out_path=None, countdown_seconds=3, model="lite"):
    cap = open_source(source)
    if cap is None:
        return 1

    writer = None
    detector = pm.PoseDetector(model=model)
    low, high = EXERCISES[exercise]["range"]
    counter = RepCounter(low, high)
    angle = None
    frames, started = 0, time.time()
    fps = 0.0

    try:
        if not countdown(cap, countdown_seconds, show):
            return 0

        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break

            img = detector.get_pose(img, draw=True)
            detector.get_position(img, draw=False)

            angle = measure(detector, img, exercise, draw=True)
            counter.update(angle)

            frames += 1
            elapsed = time.time() - started
            if elapsed > 0:
                fps = frames / elapsed
            draw_hud(img, counter.count, counter.percent, angle, exercise, fps)

            if out_path:
                if writer is None:
                    h, w = img.shape[:2]
                    src_fps = cap.get(cv.CAP_PROP_FPS)
                    # A webcam often reports 0 here. The old code divided by
                    # this value straight away and raised ZeroDivisionError.
                    if not src_fps or src_fps <= 0 or src_fps > 240:
                        src_fps = 25.0
                    writer = cv.VideoWriter(out_path,
                                            cv.VideoWriter_fourcc(*"mp4v"),
                                            src_fps, (w, h))
                writer.write(img)

            if show:
                cv.imshow("AI Trainer", img)
                if cv.waitKey(1) == 27:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        detector.close()
        if show:
            cv.destroyAllWindows()

    print(f"{EXERCISES[exercise]['label']}: {counter.reps} reps over {frames} frames "
          f"at {fps:.1f} fps")
    if out_path:
        print(f"wrote {out_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pose estimated gym rep counter")
    parser.add_argument("--exercise", choices=sorted(EXERCISES),
                        help="skip the chooser dialog")
    parser.add_argument("--source", default="0",
                        help="camera index or path to a video file (default 0)")
    parser.add_argument("--out", help="write an annotated video here")
    parser.add_argument("--model", default="lite", choices=["lite", "full", "heavy"],
                        help="pose model size (default lite)")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--headless", action="store_true",
                        help="do not open a window, for servers and CI")
    args = parser.parse_args(argv)

    exercise = args.exercise or choose_exercise()
    if exercise is None:
        print("no workout chosen")
        return 0

    return run(exercise, source=args.source, show=not args.headless,
               out_path=args.out, countdown_seconds=args.countdown, model=args.model)


if __name__ == "__main__":
    sys.exit(main())
