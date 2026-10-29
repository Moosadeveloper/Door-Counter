"""
video_processor.py
------------------
The whole pipeline in one readable function, process_video():

  PASS 1  detect people (YOLO) + track them (ByteTrack) + collect a few face
          fingerprints per track                                  [slow part]
  THEN    glue track fragments into PEOPLE (faces + movement)
          find entry / exit events at the door line
          compute dwell times
  PASS 2  draw the annotated video, convert it for the browser
  SAVE    results.json, results.csv, events.csv, face/person images

Compared with the old version this file is ~200 lines instead of 2891 and
every step lives in its own small module (identity.py, events.py, ...).
"""
import json
import os
import shutil
import time
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from collections import Counter

from config import AnalysisConfig, InstitutionConfig
from detectors import YoloByteTrackDetector
from events import analyze_person, build_results, format_time
from faces import FaceEngine, FaceModelsMissing
from geometry import DoorLine
from identity import Tracklet, resolve_identities, torso_hist
from render import render_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRACKER_YAML = os.path.join(BASE_DIR, "bytetrack_custom.yaml")
DEFAULT_YOLO = os.path.join(MODELS_DIR, "yolov8n.pt")


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:      # 0, NaN or nonsense
        fps = 25.0
    info = {"fps": float(fps),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}
    cap.release()
    if info["width"] <= 0 or info["height"] <= 0:
        raise RuntimeError("Could not read the video size.")
    return info


def _crop(frame: np.ndarray, box, max_h: int = 200) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] > max_h:
        k = max_h / crop.shape[0]
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return crop.copy()


def label_persons(persons: dict, tracklets: dict, registry, icfg: InstitutionConfig) -> None:
    """Give every PERSON found in the video a registered name - or flag them
    as 'unknown' (face seen, not registered) / 'face_hidden' (no face seen)."""
    for p in persons.values():
        vecs = [v for tid in p.track_ids for _, v in tracklets[tid].faces]
        if not vecs:
            p.flag = "face_hidden"
            continue
        votes, best = Counter(), {}
        for v in vecs:
            m = registry.match(v, icfg.recog_thr, icfg.recog_margin)
            if m is not None:
                votes[m.person_id] += 1
                best[m.person_id] = max(best.get(m.person_id, 0.0), m.score)
        cm = registry.match(p.centroid, icfg.recog_thr, icfg.recog_margin) if p.centroid is not None else None
        chosen = None
        if cm is not None and (votes[cm.person_id] >= 1 or len(vecs) == 1):
            chosen = cm.person_id
        elif votes:
            (pid, v), *rest = votes.most_common(2)
            if v >= icfg.votes_needed and v >= 2 * (rest[0][1] if rest else 0):
                chosen = pid
        if chosen is None:
            p.flag = "unknown"
            continue
        info = registry.get(chosen) or {}
        p.registered_id, p.name, p.roll_no = chosen, info.get("name"), info.get("roll_no")
        p.match_score = round(float(max(best.get(chosen, 0.0), cm.score if cm and cm.person_id == chosen else 0.0)), 3)


def process_video(input_path: str, output_dir: str, line_position: float = 0.5,
                  model_path: Optional[str] = None,
                  config: Optional[AnalysisConfig] = None,
                  progress_cb: Optional[Callable[[float, str], None]] = None,
                  detector=None, face_engine=None, registry=None,
                  inst_cfg: Optional[InstitutionConfig] = None):
    """
    Returns (summary_dict, processed_video_path).
    `line_position` is kept for backward compatibility (vertical line).
    `detector` / `face_engine` can be injected - the tests use fakes.
    """
    t_start = time.time()
    cfg = (config or AnalysisConfig(line=(line_position, 0.0, line_position, 1.0))).normalise()

    def report(p: float, stage: str) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(1.0, p)), stage)

    os.makedirs(output_dir, exist_ok=True)
    info = probe_video(input_path)
    fps, W, H, total = info["fps"], info["width"], info["height"], info["frames"]
    door = DoorLine(cfg.line, W, H, cfg.flip, cfg.zone_frac)
    warnings = []

    # ---- face engine (optional) ------------------------------------------
    if not cfg.use_faces:
        face_engine = None
    elif face_engine is None:
        try:
            face_engine = FaceEngine(MODELS_DIR, cfg.face_det_score, cfg.min_face_px)
        except FaceModelsMissing as e:
            warnings.append(f"Face recognition is OFF: {e} IDs use movement only.")
        except Exception as e:                     # e.g. old OpenCV build
            warnings.append(f"Face recognition is OFF: {e}")

    if detector is None:
        model = model_path or (DEFAULT_YOLO if os.path.exists(DEFAULT_YOLO) else "yolov8n.pt")
        detector = YoloByteTrackDetector(model, TRACKER_YAML, cfg.conf, cfg.iou, cfg.imgsz, cfg.stride)

    # =======================================================================
    # PASS 1: detect + track + collect face fingerprints
    # =======================================================================
    tracklets = {}
    face_every = max(1, int(round(cfg.face_every_seconds * fps)))
    color_every = max(1, int(round(fps)))
    last_frame = -1
    report(0.0, "Detecting and tracking people")

    for frame_idx, frame, dets in detector.run(input_path):
        last_frame = frame_idx
        for d in dets:
            tl = tracklets.get(d.track_id)
            if tl is None:
                tl = tracklets[d.track_id] = Tracklet(d.track_id)
            tl.add(frame_idx, d.box)

            x1, y1, x2, y2 = d.box
            area, bh = (x2 - x1) * (y2 - y1), y2 - y1

            if area > 0 and (tl.snapshot is None or area > 1.25 * tl.snapshot[0]):
                crop = _crop(frame, d.box)
                if crop is not None:
                    tl.snapshot = (area, crop)

            if frame_idx - tl.last_color_try >= color_every:
                tl.last_color_try = frame_idx
                hist = torso_hist(frame, d.box)
                if hist is not None:
                    tl.add_color(hist)

            if (face_engine is not None and bh >= cfg.min_person_height_px
                    and len(tl.faces) < cfg.max_faces_per_track
                    and frame_idx - tl.last_face_try >= face_every):
                tl.last_face_try = frame_idx
                sample = face_engine.extract(frame, d.box, d.meta)
                if sample is not None:
                    tl.add_face(sample.quality, sample.vec, sample.thumb_jpg, cfg.max_faces_per_track)

        if total:
            report(0.75 * min(1.0, (frame_idx + 1) / total), "Detecting and tracking people")

    if last_frame < 0:
        raise RuntimeError("No frames could be read from this video.")
    end_frame = max(total, last_frame + 1)

    # =======================================================================
    # WHO IS WHO + DOOR EVENTS
    # =======================================================================
    report(0.76, "Matching people (faces + movement)")
    ident = resolve_identities(tracklets, door, fps, cfg)
    persons = ident.persons
    if registry is not None and registry.count():
        report(0.78, "Matching faces with registered people")
        label_persons(persons, tracklets, registry, inst_cfg or InstitutionConfig())

    confirm_n = max(2, int(round(cfg.confirm_seconds * fps / cfg.stride)))
    span = (-cfg.span_margin, 1.0 + cfg.span_margin)
    timelines = []
    for pid, p in persons.items():
        obs = []
        for tid in p.track_ids:
            obs += tracklets[tid].obs_series(door, cfg.anchor)
        tl = analyze_person(pid, obs, door.zone, confirm_n, span, int(cfg.interp_gap_seconds * fps))
        if tl is not None:
            timelines.append(tl)

    report(0.80, "Counting entries and exits")
    summary = build_results(timelines, persons, fps, end_frame, cfg.face_match_thr, cfg.capacity)

    # ---- save person / face images ---------------------------------------
    for sub in ("faces", "people"):
        shutil.rmtree(os.path.join(output_dir, sub), ignore_errors=True)
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    for pid, p in persons.items():
        if p.best_face_jpg:
            with open(os.path.join(output_dir, "faces", f"person_{pid}.jpg"), "wb") as fh:
                fh.write(p.best_face_jpg)
        if p.snapshot_jpg:
            with open(os.path.join(output_dir, "people", f"person_{pid}.jpg"), "wb") as fh:
                fh.write(p.snapshot_jpg)

    # =======================================================================
    # PASS 2: annotated video
    # =======================================================================
    out_video = os.path.join(output_dir, "processed_video.mp4")
    playable = False
    try:
        def render_progress(x: float) -> None:
            report(0.80 + 0.17 * x, "Drawing the annotated video")

        playable, _ = render_video(input_path, out_video, fps, (W, H), tracklets,
                                   ident.person_of_track, timelines, persons, door,
                                   render_progress, end_frame, cfg.stride, cfg.capacity)
        report(0.99, "Saving results")
        if not playable:
            warnings.append("ffmpeg was not found, so the annotated video may not play in the browser. "
                            "Run: pip install imageio-ffmpeg")
    except Exception as e:
        warnings.append(f"Could not create the annotated video: {e}")

    # ---- warnings that help the user understand the result -----------------
    if face_engine is not None and not any(p.face_samples for p in persons.values()):
        warnings.append("No clear faces were found. IDs rely on movement only. Faces work best when "
                        "people face the camera and are at least ~80 px tall.")
    if not persons:
        warnings.append("No people were detected. Check the video and try a lower confidence or larger image size.")

    # =======================================================================
    # SAVE
    # =======================================================================
    summary.update({
        "video": {**info, "frames": end_frame, "duration_seconds": round(end_frame / fps, 2),
                  "duration": format_time(end_frame / fps), "browser_playable": playable},
        "line": {"x1": cfg.line[0], "y1": cfg.line[1], "x2": cfg.line[2], "y2": cfg.line[3],
                 "flip": cfg.flip},
        "identity": ident.stats,
        "warnings": warnings,
        "processing": {"seconds": round(time.time() - t_start, 1),
                       "face_engine": getattr(face_engine, "name", None),
                       "stride": cfg.stride},
        "config": cfg.to_dict(),
    })

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    cols = ["person_id", "entry_time", "exit_time", "dwell_time", "status", "entries", "exits",
            "first_seen", "identity", "face_samples"]
    pd.DataFrame(summary["records"], columns=cols).to_csv(
        os.path.join(output_dir, "results.csv"), index=False)
    ev_cols = ["time", "person_id", "type", "face_verified", "similarity", "inferred", "dwell_time"]
    pd.DataFrame(summary["events"], columns=ev_cols).to_csv(
        os.path.join(output_dir, "events.csv"), index=False)

    report(1.0, "Done")
    return summary, out_video
