"""
live.py
-------
LIVE camera mode for the institution:

    camera frame -> YOLO finds people -> ByteTrack keeps an ID per person
                 -> every ~0.25 s look at each person's face
                 -> compare with REGISTERED faces (registry.py)
                 -> 2 matching looks  = recognised  -> attendance IN / OUT
                 -> face visible but matches nobody -> "UNKNOWN" alert + photo
                 -> walks in, face never visible    -> "FACE HIDDEN" alert + photo

Three camera modes
  entry  camera at the gate facing people coming IN  -> recognised = checked in
  exit   camera facing people going OUT             -> recognised = checked out
  door   one camera sees both sides of a door line  -> direction decides in / out

Split in two parts so it can be tested without a camera:
  LiveEngine   pure logic: process(frame, detections, time) -> annotated frame
  LiveSession  thread that reads the camera / RTSP / video file and runs YOLO
"""
import os
import re
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import InstitutionConfig
from geometry import DoorLine

# CCTV / RTSP: use TCP instead of UDP (UDP drops packets on Wi-Fi / busy networks
# and the picture turns grey or smeared). Must be set before the stream opens.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def mask_url(text: str) -> str:
    """rtsp://admin:secret@1.2.3.4/... -> rtsp://admin:****@1.2.3.4/...  (never show passwords)"""
    return re.sub(r"(://[^:/@\s]+:)\S*@", r"\1****@", str(text or ""))


GREEN = (80, 200, 60)
RED = (40, 40, 230)
ORANGE = (0, 140, 255)
GRAY = (200, 200, 200)
YELLOW = (0, 230, 255)


# ---------------------------------------------------------------------------
# per-person (per-track) memory
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    track_id: int
    first_ts: float
    last_ts: float
    first_h: float
    max_h: float = 0.0
    face_tries: int = 0
    faces: int = 0
    nomatch: int = 0
    last_face_try: float = -1e9
    votes: Counter = field(default_factory=Counter)
    best_score: Dict[str, float] = field(default_factory=dict)
    vecs: List[np.ndarray] = field(default_factory=list)
    best_face: Optional[Tuple[float, bytes]] = None       # (quality, jpg)
    best_body: Optional[Tuple[float, bytes]] = None       # (area, jpg)
    identity: Optional[str] = None                        # registered person_id
    score: float = 0.0
    status: str = "checking"                              # checking | known | unknown | face_hidden
    alert_id: Optional[int] = None
    # door-line mode
    side: Optional[int] = None
    cand: Optional[int] = None
    cand_n: int = 0
    pending: List[Tuple[str, datetime]] = field(default_factory=list)
    crossed_in: bool = False


def _jpg(img: np.ndarray, q: int = 85) -> Optional[bytes]:
    if img is None or img.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None


def _text(img, s, org, scale, color, thick, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 3, y - h - 5), (x + w + 3, y + base + 1), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# the logic
# ---------------------------------------------------------------------------
class LiveEngine:
    def __init__(self, registry, attendance, face_engine, cfg: InstitutionConfig,
                 camera: str = "Gate", mode: str = "entry",
                 line: Optional[Tuple[float, float, float, float]] = None, flip: bool = False,
                 on_event: Optional[Callable[[dict], None]] = None):
        if mode not in ("entry", "exit", "door"):
            raise ValueError("mode must be entry, exit or door")
        if mode == "door" and not line:
            raise ValueError("Door mode needs a door line.")
        self.reg, self.att, self.faces, self.cfg = registry, attendance, face_engine, cfg
        self.camera, self.mode, self.line, self.flip = camera, mode, line, flip
        self.on_event = on_event
        self.tracks: Dict[int, TrackState] = {}
        self.door: Optional[DoorLine] = None
        self.feed: deque = deque(maxlen=200)
        self._present_cache = (0.0, 0)

    # ------------------------------------------------------------ events
    def _emit(self, ev: dict) -> None:
        ev.setdefault("camera", self.camera)
        self.feed.appendleft(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                traceback.print_exc()

    def _mark(self, st: TrackState, kind: str, when: datetime) -> bool:
        person = self.reg.get(st.identity)
        if person is None:
            return False
        face = st.best_face[1] if st.best_face else (st.best_body[1] if st.best_body else None)
        ev = self.att.mark(kind, person, when, self.camera, st.score, face)
        if ev is not None:
            ev["type"] = kind
            self._emit(ev)
            self._present_cache = (0.0, 0)
            return True
        return False

    def _alert(self, st: TrackState, kind: str, when: datetime) -> None:
        a = self.att.add_alert(kind, when, self.camera,
                               st.best_body[1] if st.best_body else None,
                               st.best_face[1] if st.best_face else None)
        st.alert_id, st.status = a["id"], kind
        a["type"] = kind
        self._emit(a)

    # ------------------------------------------------------------ identity
    def _try_identify(self, st: TrackState) -> None:
        if not st.votes:
            return
        (pid, v), *rest = st.votes.most_common(2)
        other = rest[0][1] if rest else 0
        ok = v >= self.cfg.votes_needed and v >= 2 * other
        if not ok and len(st.vecs) >= 3 and v >= 1:              # average of several looks agrees
            c = np.mean(st.vecs, axis=0)
            c /= max(1e-9, float(np.linalg.norm(c)))
            m = self.reg.match(c, self.cfg.recog_thr, self.cfg.recog_margin)
            ok = m is not None and m.person_id == pid
        if not ok:
            return
        # the same registered person cannot be two people on screen at once
        for other_st in self.tracks.values():
            if other_st is not st and other_st.identity == pid:
                if other_st.score >= st.best_score.get(pid, 0):
                    st.votes.pop(pid, None)
                    return
                other_st.identity, other_st.status = None, "checking"
                other_st.votes.pop(pid, None)
        st.identity, st.score, st.status = pid, st.best_score.get(pid, 0.0), "known"

    def _on_identified(self, st: TrackState, when: datetime) -> None:
        person = self.reg.get(st.identity) or {}
        if st.alert_id is not None:                               # alert raised earlier -> now known
            self.att.resolve_alert(st.alert_id, person.get("name", ""))
        logged = False
        if self.mode == "entry":
            logged = self._mark(st, "in", when)
        elif self.mode == "exit":
            logged = self._mark(st, "out", when)
        else:
            for kind, t in st.pending:
                logged = self._mark(st, kind, t) or logged
            st.pending.clear()
        if logged:
            return
        self._emit({"type": "recognized", "time": when.strftime("%H:%M:%S"), "person_id": st.identity,
                    "name": person.get("name"), "roll_no": person.get("roll_no"),
                    "score": round(float(st.score), 3)})

    # ------------------------------------------------------------ door line
    def _door_step(self, st: TrackState, box, when: datetime) -> None:
        x1, y1, x2, y2 = box
        d, t = self.door.locate((x1 + x2) / 2.0, y2)             # feet position
        s = 1 if d > self.door.zone else (-1 if d < -self.door.zone else 0)
        if s == 0 or s == st.side:
            st.cand, st.cand_n = None, 0
            return
        if s != st.cand:
            st.cand, st.cand_n = s, 1
        else:
            st.cand_n += 1
        need = 2 if st.side is None else self.cfg.confirm_frames
        if st.cand_n < need:
            return
        prev, st.side, st.cand, st.cand_n = st.side, s, None, 0
        if prev is None or not (-0.25 <= t <= 1.25):
            return
        kind = "in" if s == 1 else "out"
        if kind == "in":
            st.crossed_in = True
        if st.identity is not None:
            self._mark(st, kind, when)
        else:
            st.pending.append((kind, when))

    # ------------------------------------------------------------ main step
    def process(self, frame: np.ndarray, dets: list, now: Optional[datetime] = None,
                draw: bool = True) -> np.ndarray:
        now = now or datetime.now()
        t = now.timestamp()
        H, W = frame.shape[:2]
        if self.mode == "door" and self.door is None:
            self.door = DoorLine(self.line, W, H, self.flip)
        cfg = self.cfg
        seen = set()

        for d in dets:
            x1, y1, x2, y2 = d.box
            bh, area = y2 - y1, max(0.0, (x2 - x1) * (y2 - y1))
            st = self.tracks.get(d.track_id)
            if st is None:
                st = self.tracks[d.track_id] = TrackState(d.track_id, t, t, bh)
            st.last_ts = t
            st.max_h = max(st.max_h, bh)
            seen.add(d.track_id)

            if st.best_body is None or area > 1.2 * st.best_body[0]:
                crop = frame[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))]
                jb = _jpg(crop)
                if jb:
                    st.best_body = (area, jb)

            if self.door is not None:
                self._door_step(st, d.box, now)

            # ---- look at the face ------------------------------------------
            if (st.identity is None and self.faces is not None and self.reg.count()
                    and bh >= 80 and t - st.last_face_try >= cfg.face_every_seconds):
                st.last_face_try = t
                st.face_tries += 1
                sample = self.faces.extract(frame, d.box, getattr(d, "meta", None))
                if sample is not None:
                    st.faces += 1
                    st.vecs = (st.vecs + [sample.vec])[-12:]
                    if sample.thumb_jpg and (st.best_face is None or sample.quality > st.best_face[0]):
                        st.best_face = (sample.quality, sample.thumb_jpg)
                    m = self.reg.match(sample.vec, cfg.recog_thr, cfg.recog_margin)
                    if m is not None:
                        st.votes[m.person_id] += 1
                        st.best_score[m.person_id] = max(st.best_score.get(m.person_id, 0.0), m.score)
                    else:
                        st.nomatch += 1
                    self._try_identify(st)
                    if st.identity is not None:
                        self._on_identified(st, now)

            # ---- alerts -------------------------------------------------------
            if st.identity is None and st.alert_id is None and self.reg.count():
                age = t - st.first_ts
                relevant = st.crossed_in if self.mode == "door" else True
                approaching = st.max_h >= 1.08 * st.first_h or self.mode == "door"
                if (relevant and st.faces >= cfg.min_faces_for_unknown and age >= cfg.unknown_after_seconds
                        and st.nomatch >= 0.6 * st.faces):
                    self._alert(st, "unknown", now)
                elif (relevant and approaching and st.faces == 0 and age >= cfg.hidden_after_seconds
                      and st.max_h >= cfg.min_alert_height_px and st.face_tries >= 6):
                    self._alert(st, "face_hidden", now)

        # ---- forget people who left the picture ----------------------------
        for tid in [k for k, s in self.tracks.items()
                    if k not in seen and t - s.last_ts > cfg.forget_track_seconds]:
            self.tracks.pop(tid, None)

        if draw:
            self._draw(frame, dets, now)
        return frame

    # ------------------------------------------------------------ drawing
    def present_today(self, now: datetime) -> int:
        ts, n = self._present_cache
        if now.timestamp() - ts > 5:
            try:
                rep = self.att.day_report(self.reg.list(), now.strftime("%Y-%m-%d"))
                n = rep["present"]
            except Exception:
                n = 0
            self._present_cache = (now.timestamp(), n)
        return n

    def _draw(self, frame: np.ndarray, dets: list, now: datetime) -> None:
        H, W = frame.shape[:2]
        s = max(0.55, min(1.5, H / 720.0))
        th = max(1, int(round(2 * s)))
        if self.door is not None:
            p1 = tuple(int(v) for v in self.door.p1)
            p2 = tuple(int(v) for v in self.door.p2)
            cv2.line(frame, p1, p2, YELLOW, max(2, int(3 * s)), cv2.LINE_AA)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            nx, ny = self.door.inside_normal()
            tip = (int(mx + nx * 60 * s), int(my + ny * 60 * s))
            cv2.arrowedLine(frame, (mx, my), tip, YELLOW, max(2, int(3 * s)), cv2.LINE_AA, tipLength=0.3)
            _text(frame, "IN", (tip[0] + 5, tip[1] + 5), 0.6 * s, YELLOW, th)

        for d in dets:
            st = self.tracks.get(d.track_id)
            if st is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in d.box)
            if st.status == "known":
                p = self.reg.get(st.identity) or {}
                color, label = GREEN, f"{p.get('name', '?')} ({p.get('roll_no', '')})"
            elif st.status == "unknown":
                color, label = RED, "UNKNOWN PERSON"
            elif st.status == "face_hidden":
                color, label = ORANGE, "FACE HIDDEN"
            else:
                color, label = GRAY, "checking..."
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, th + (1 if st.status != "checking" else 0))
            _text(frame, label, (x1, max(int(20 * s), y1 - 8)), 0.6 * s, (255, 255, 255), th, bg=color)

        banner = f"{self.camera}  |  {self.mode.upper()} camera  |  {now.strftime('%d %b %Y  %H:%M:%S')}"
        _text(frame, banner, (10, int(26 * s)), 0.6 * s, (255, 255, 255), th)
        if self.reg.count():
            _text(frame, f"Present today: {self.present_today(now)} / {self.reg.count()}",
                  (10, int(56 * s)), 0.6 * s, GREEN, th)
        else:
            _text(frame, "No one registered yet - add people on the People page",
                  (10, int(56 * s)), 0.6 * s, ORANGE, th)


# ---------------------------------------------------------------------------
# YOLO + ByteTrack for single frames
# ---------------------------------------------------------------------------
class YoloLiveTracker:
    def __init__(self, model_path: str, tracker_yaml: str, conf: float, iou: float, imgsz: int):
        from ultralytics import YOLO                     # heavy import, only when live mode starts
        from detectors import Detection
        self._Detection = Detection
        self.model = YOLO(model_path)
        self.tracker_yaml, self.conf, self.iou, self.imgsz = tracker_yaml, conf, iou, imgsz

    def update(self, frame: np.ndarray) -> list:
        r = self.model.track(frame, persist=True, tracker=self.tracker_yaml, classes=[0],
                             conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        out = []
        b = r.boxes
        if b is not None and b.id is not None:
            for tid, box, c in zip(b.id.int().cpu().tolist(), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                out.append(self._Detection(int(tid), tuple(float(v) for v in box), float(c)))
        return out


def parse_source(source: str):
    """'0' / '1' -> webcam index, anything else -> file path or rtsp/http URL."""
    s = (source or "").strip()
    if s.isdigit():
        return int(s)
    return s


# ---------------------------------------------------------------------------
# the running camera
# ---------------------------------------------------------------------------
class LiveSession:
    def __init__(self, source: str, engine: LiveEngine, tracker_factory: Callable[[], object],
                 label: str = ""):
        self.source_text = source
        self.label = label or source
        self.source = parse_source(source)
        self.engine = engine
        self.tracker_factory = tracker_factory
        self.state = "starting"          # starting | running | stopped | error | finished
        self.error = ""
        self.fps = 0.0
        self.reconnects = 0
        self.started = datetime.now()
        self._stop = threading.Event()
        self._jpeg: Optional[bytes] = None
        self._raw: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------ control
    def start(self) -> "LiveSession":
        self._thread.start()
        return self

    def stop(self, wait: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(wait)
        if self.state in ("starting", "running"):
            self.state = "stopped"

    @property
    def running(self) -> bool:
        return self.state in ("starting", "running", "reconnecting")

    def latest(self) -> Tuple[int, Optional[bytes]]:
        with self._lock:
            return self._seq, self._jpeg

    def latest_raw(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "fps": round(self.fps, 1),
                "source": self.label, "camera": self.engine.camera, "mode": self.engine.mode,
                "started": self.started.isoformat(timespec="seconds"),
                "people_on_screen": len(self.engine.tracks), "reconnects": self.reconnects,
                "line": list(self.engine.line) if self.engine.line else None, "flip": self.engine.flip}

    # ------------------------------------------------------------ loop
    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # keep only fresh frames (CCTV lag)
        except Exception:
            pass
        return cap

    def _handle(self, frame, tracker, counters) -> None:
        raw = frame.copy()
        dets = tracker.update(frame)
        out = self.engine.process(frame, dets, datetime.now())
        jpg = _jpg(out, 75)
        with self._lock:
            self._raw, self._jpeg = raw, jpg
            self._seq += 1
        counters[0] += 1
        if time.time() - counters[1] >= 1.0:
            self.fps = counters[0] / (time.time() - counters[1])
            counters[0], counters[1] = 0, time.time()

    def _run(self) -> None:
        cap = None
        try:
            is_file = isinstance(self.source, str) and os.path.isfile(self.source)
            cap = self._open()
            if cap is None:
                raise RuntimeError(f"Could not open the camera / stream '{mask_url(self.source_text)}'. "
                                   "Check the IP address, user name, password and channel, that the "
                                   "computer is on the same network as the CCTV/DVR, and that no other "
                                   "app is using the camera.")
            tracker = self.tracker_factory()
            counters = [0, time.time()]

            if is_file:                                          # ---- recorded video (testing)
                file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                if not (1 <= file_fps <= 240):
                    file_fps = 25.0
                self.state = "running"
                t0, frame_i = time.time(), 0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self.state = "finished"
                        break
                    frame_i += 1
                    # play the file at real speed (so times look like a real camera)
                    wait = t0 + frame_i / file_fps - time.time()
                    if wait > 0:
                        time.sleep(min(wait, 0.2))
                    elif wait < -0.5:                            # too slow: skip frames to keep up
                        for _ in range(min(int(-wait * file_fps), 10)):
                            cap.grab()
                            frame_i += 1
                    self._handle(frame, tracker, counters)
                return

            # ---- live camera / CCTV: keeps going, reconnects by itself if the network drops
            while not self._stop.is_set():
                grab = _Grabber(cap)
                grab.start()
                self.state, self.error = "running", ""
                misses = 0
                while not self._stop.is_set():
                    frame = grab.read(timeout=2.0)
                    if frame is None:
                        misses += 1
                        if misses > 5 or grab.dead:
                            break
                        continue
                    misses = 0
                    self._handle(frame, tracker, counters)
                grab.stop()
                cap.release()
                cap = None
                if self._stop.is_set():
                    break
                self.state = "reconnecting"
                self.error = "Connection to the camera was lost - trying again every few seconds..."
                self.fps = 0.0
                while not self._stop.is_set():
                    self._stop.wait(3.0)
                    if self._stop.is_set():
                        break
                    cap = self._open()
                    if cap is not None:
                        self.reconnects += 1
                        break
        except Exception as e:
            traceback.print_exc()
            self.error = mask_url(str(e) or e.__class__.__name__)
            self.state = "error"
        finally:
            if cap is not None:
                cap.release()
            if self.state in ("starting", "running", "reconnecting"):
                self.state = "stopped"


class _Grabber:
    """Reads a live camera in its own thread and keeps only the NEWEST frame,
    so a slow computer shows 'now' instead of falling further and further behind."""

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._new = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.dead = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(2.0)

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            ok, fr = self.cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    self.dead = True
                    return
                time.sleep(0.05)
                continue
            fails = 0
            with self._lock:
                self._frame = fr
            self._new.set()

    def read(self, timeout: float = 1.0):
        if not self._new.wait(timeout):
            return None
        with self._lock:
            fr, self._frame = self._frame, None
            self._new.clear()
        return fr


def grab_one_frame(source: str, timeout_frames: int = 30) -> Optional[np.ndarray]:
    """One picture from a camera / stream / file (for drawing the door line)."""
    cap = cv2.VideoCapture(parse_source(source))
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(timeout_frames):
            ok, fr = cap.read()
            if not ok:
                break
            frame = fr
            if fr.mean() > 8:
                break
        return frame
    finally:
        cap.release()
