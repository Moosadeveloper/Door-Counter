"""
Test doubles: a scripted "scene" video, a fake detector that reports the
scripted boxes (with deliberately messy tracker IDs), and a fake face engine.
Lets us test the WHOLE pipeline without YOLO / face models / internet.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors import Detection      # noqa: E402
from faces import FaceSample         # noqa: E402


@dataclass
class Actor:
    name: str
    color: Tuple[int, int, int]
    waypoints: List[Tuple[int, float, float]]            # (frame, x_center, y_foot)
    track_segments: List[Tuple[int, int, int]]           # (first_frame, last_frame, tracker_id)
    face_frames: Optional[List[Tuple[int, int]]] = None  # frames where the face is visible (None = always)
    size: Tuple[int, int] = (40, 100)

    def pos(self, f: int):
        wp = self.waypoints
        if f < wp[0][0] or f > wp[-1][0]:
            return None
        for (f0, x0, y0), (f1, x1, y1) in zip(wp, wp[1:]):
            if f0 <= f <= f1:
                a = 0 if f1 == f0 else (f - f0) / (f1 - f0)
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
        return wp[-1][1], wp[-1][2]

    def track_id(self, f: int):
        for a, b, tid in self.track_segments:
            if a <= f <= b:
                return tid
        return None

    def face_visible(self, f: int) -> bool:
        if self.face_frames is None:
            return True
        return any(a <= f <= b for a, b in self.face_frames)


@dataclass
class Scene:
    actors: List[Actor]
    frames: int = 320
    fps: int = 10
    size: Tuple[int, int] = (640, 360)


def write_scene_video(path: str, scene: Scene) -> None:
    W, H = scene.size
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), scene.fps, (W, H))
    for f in range(scene.frames):
        img = np.full((H, W, 3), 90, np.uint8)
        for a in scene.actors:
            p = a.pos(f)
            if p:
                x, yf = p
                w, h = a.size
                cv2.rectangle(img, (int(x - w / 2), int(yf - h)), (int(x + w / 2), int(yf)), a.color, -1)
        wr.write(img)
    wr.release()


class FakeDetector:
    def __init__(self, scene: Scene):
        self.scene = scene

    def run(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = []
            for a in self.scene.actors:
                p, tid = a.pos(f), a.track_id(f)
                if p is None or tid is None:
                    continue
                x, yf = p
                w, h = a.size
                dets.append(Detection(tid, (x - w / 2, yf - h, x + w / 2, yf), 0.9,
                                      {"gt": a.name, "face": a.face_visible(f)}))
            yield f, frame, dets
            f += 1
        cap.release()


class FakeFaceEngine:
    """Every person 'name' has a fixed random 128-d fingerprint (+ a little noise per sighting)."""
    name = "fake-face-engine"

    def __init__(self, seed: int = 0, noise: float = 0.05):
        self.rng = np.random.RandomState(seed)
        self.noise = noise
        self.base = {}

    def _vec(self, name: str) -> np.ndarray:
        if name not in self.base:
            v = self.rng.randn(128).astype(np.float32)
            self.base[name] = v / np.linalg.norm(v)
        return self.base[name]

    def extract(self, frame, box, meta=None):
        if not meta or not meta.get("face"):
            return None
        v = self._vec(meta["gt"]) + self.noise * self.rng.randn(128).astype(np.float32)
        v /= np.linalg.norm(v)
        thumb = np.full((32, 32, 3), 200, np.uint8)
        ok, buf = cv2.imencode(".jpg", thumb)
        return FaceSample(vec=v, quality=0.9, thumb_jpg=buf.tobytes())
