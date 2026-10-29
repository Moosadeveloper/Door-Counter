"""
detectors.py
------------
Wraps YOLOv8 (person detection) + ByteTrack (tracking) behind one tiny
interface, so the rest of the pipeline - and the tests - do not care which
detector is used:

    for frame_idx, frame_bgr, detections in detector.run(video_path): ...
"""
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    track_id: int
    box: Tuple[float, float, float, float]      # x1, y1, x2, y2 in pixels
    conf: float = 1.0
    meta: Optional[dict] = None                 # only used by test fakes


class YoloByteTrackDetector:
    def __init__(self, model_path: str, tracker_yaml: str, conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 640, stride: int = 1):
        self.model_path = model_path
        self.tracker_yaml = tracker_yaml
        self.conf, self.iou, self.imgsz, self.stride = conf, iou, imgsz, stride

    def run(self, video_path: str) -> Iterator[Tuple[int, np.ndarray, List[Detection]]]:
        from ultralytics import YOLO        # imported here: heavy, and not needed for tests

        model = YOLO(self.model_path)       # fresh model per video = fresh tracker state
        results = model.track(
            source=video_path,
            tracker=self.tracker_yaml,      # absolute path: works from any folder
            persist=True,
            stream=True,
            classes=[0],                    # 0 = person
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            vid_stride=self.stride,
            verbose=False,
        )
        for i, result in enumerate(results):
            dets: List[Detection] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for tid, box, c in zip(ids, xyxy, confs):
                    dets.append(Detection(int(tid), tuple(float(v) for v in box), float(c)))
            yield i * self.stride, result.orig_img, dets
