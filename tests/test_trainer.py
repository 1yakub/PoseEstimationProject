"""Checks that run without a camera, so CI and a headless box can run them.

    pip install pytest
    pytest -q
"""

import os
import sys

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AITrainer  # noqa: E402
import PoseModule as pm  # noqa: E402

DEMO_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "AITrainer", "aitrainer-demo.jpg")


def curl_cycle(reps):
    """Angles for a full curl: extended arm, flexed, extended again."""
    angles = []
    for _ in range(reps):
        angles += list(np.linspace(170, 40, 12))
        angles += list(np.linspace(40, 170, 12))
    return angles


class TestRepCounter:
    def test_counts_full_reps(self):
        low, high = AITrainer.EXERCISES["curl"]["range"]
        counter = AITrainer.RepCounter(low, high)
        for angle in curl_cycle(3):
            counter.update(angle)
        assert counter.reps == 3

    def test_half_rep_does_not_count(self):
        low, high = AITrainer.EXERCISES["curl"]["range"]
        counter = AITrainer.RepCounter(low, high)
        for angle in np.linspace(170, 40, 12):  # go up, never come back
            counter.update(angle)
        assert counter.reps == 0

    def test_shallow_rep_does_not_count(self):
        """Curling only halfway must not be rewarded with a rep."""
        low, high = AITrainer.EXERCISES["curl"]["range"]
        counter = AITrainer.RepCounter(low, high)
        for angle in list(np.linspace(170, 110, 12)) + list(np.linspace(110, 170, 12)):
            counter.update(angle)
        assert counter.reps == 0

    def test_missing_angle_is_ignored(self):
        low, high = AITrainer.EXERCISES["curl"]["range"]
        counter = AITrainer.RepCounter(low, high)
        for _ in range(50):
            counter.update(None)
        assert counter.reps == 0

    def test_percent_tracks_range(self):
        counter = AITrainer.RepCounter(50, 160)
        counter.update(160)
        assert counter.percent == pytest.approx(0, abs=1)
        counter.update(50)
        assert counter.percent == pytest.approx(100, abs=1)


class TestPoseDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        d = pm.PoseDetector(static_image_mode=True)
        yield d
        d.close()

    def test_calls_before_detection_are_safe(self, detector):
        """The old module raised AttributeError here."""
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        assert detector.get_position(img) == []
        assert detector.get_angle(img, 11, 13, 15, draw=False) is None

    def test_none_frame_is_safe(self, detector):
        assert detector.get_pose(None) is None

    def test_blank_frame_finds_nobody(self, detector):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        detector.get_pose(img, draw=True)
        assert detector.get_position(img, draw=False) == []

    @pytest.mark.skipif(not os.path.exists(DEMO_IMAGE), reason="demo image missing")
    def test_finds_a_body_in_the_demo_image(self, detector):
        img = cv.imread(DEMO_IMAGE)
        detector.get_pose(img, draw=False)
        landmarks = detector.get_position(img, draw=False)
        assert len(landmarks) > 15
        ids = {lid for lid, _, _ in landmarks}
        assert {pm.L_SHOULDER, pm.L_ELBOW, pm.L_WRIST} <= ids

    @pytest.mark.skipif(not os.path.exists(DEMO_IMAGE), reason="demo image missing")
    def test_elbow_angle_is_plausible(self, detector):
        img = cv.imread(DEMO_IMAGE)
        detector.get_pose(img, draw=False)
        detector.get_position(img, draw=False)
        angle = detector.get_angle(img, pm.L_SHOULDER, pm.L_ELBOW, pm.L_WRIST, draw=False)
        assert angle is not None
        assert 0 <= angle <= 180

    def test_angle_maths_is_right(self, detector):
        """A clean right angle must read 90 degrees."""
        detector.lmList = [[0, 100, 100], [1, 100, 200], [2, 200, 200]]
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        assert detector.get_angle(img, 0, 1, 2, draw=False) == pytest.approx(90, abs=0.5)

    def test_straight_limb_is_180(self, detector):
        detector.lmList = [[0, 100, 100], [1, 100, 200], [2, 100, 300]]
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        assert detector.get_angle(img, 0, 1, 2, draw=False) == pytest.approx(180, abs=0.5)


class TestVideoPipeline:
    def test_missing_source_fails_cleanly(self):
        assert AITrainer.open_source("no-such-file.mp4") is None

    @pytest.mark.skipif(not os.path.exists(DEMO_IMAGE), reason="demo image missing")
    def test_runs_headless_and_writes_output(self, tmp_path):
        clip = str(tmp_path / "clip.mp4")
        out = str(tmp_path / "out.mp4")
        frame = cv.imread(DEMO_IMAGE)
        h, w = frame.shape[:2]
        writer = cv.VideoWriter(clip, cv.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for _ in range(20):
            writer.write(frame)
        writer.release()

        code = AITrainer.run("curl", source=clip, show=False, out_path=out,
                             countdown_seconds=0)
        assert code == 0
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_hud_scales_to_small_frames(self):
        """The old HUD drew a bar from y=300 to y=600, off a 480p frame."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        AITrainer.draw_hud(img, 5, 50, 90, "curl", 30)
        # something must be painted inside the visible frame
        assert img.any()
        bottom_band = img[int(480 * 0.25):int(480 * 0.85), :]
        assert bottom_band.any()
