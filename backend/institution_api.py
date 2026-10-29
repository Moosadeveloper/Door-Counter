"""
institution_api.py
------------------
HTTP endpoints for the institution part (people, live camera, attendance, alerts).

  GET    /api/people                         everyone registered
  POST   /api/people                         register (form: name, roll_no, role, department,
                                             files: front, left, right, extra...)
  POST   /api/people/{pid}/photos            add more photos to someone
  PATCH  /api/people/{pid}                   change name / roll no / role / department
  DELETE /api/people/{pid}                   remove someone (and their face data)
  GET    /api/people/{pid}/image/{file}      a stored photo / face

  POST   /api/live/start                     {source, camera, mode, line, flip, video_id}
  POST   /api/live/stop
  GET    /api/live/status
  GET    /api/live/stream                    MJPEG video (use as <img src>)
  GET    /api/live/feed                      latest recognitions / alerts
  GET    /api/live/probe?source=...          one still picture (to draw the door line)

  GET    /api/attendance?day=YYYY-MM-DD      present / absent for everyone
  GET    /api/attendance.csv?day=...
  GET    /api/attendance/days                days that have data
  GET    /api/log?day=...                    every IN / OUT event
  POST   /api/attendance/manual              {person_id, kind: in|out}   (teacher override)

  GET    /api/alerts?day=...
  POST   /api/alerts/{id}/review             {reviewed: true|false}
  GET    /api/snapshot/{day}/{file}          photo from a log entry or alert
"""
import os
import threading
import time
from typing import List, Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from attendance import AttendanceDB, valid_day
from config import InstitutionConfig
from faces import FaceEngine, models_available
from live import LiveEngine, LiveSession, YoloLiveTracker, grab_one_frame, mask_url
from registry import EnrollError, PersonRegistry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DOORLOG_DATA", os.path.join(BASE_DIR, "data"))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TRACKER_YAML = os.path.join(BASE_DIR, "bytetrack_custom.yaml")
DEFAULT_YOLO = os.path.join(MODELS_DIR, "yolov8n.pt")
MAX_PHOTO_BYTES = 15 * 1024 * 1024

ICFG = InstitutionConfig()
REGISTRY = PersonRegistry(os.path.join(DATA_DIR, "registry"),
                          max_px=ICFG.registry_max_px, quality=ICFG.registry_quality)
ATTENDANCE = AttendanceDB(os.path.join(DATA_DIR, "attendance.db"),
                          os.path.join(DATA_DIR, "snapshots"), ICFG.log_cooldown_seconds,
                          snap_max_px=ICFG.snap_max_px, snap_quality=ICFG.snap_quality,
                          snap_known=ICFG.snap_known, snap_keep_days=ICFG.snap_keep_days)
ATTENDANCE.cleanup_old(force=True)                    # server start par purani photos saaf

router = APIRouter()

_face_lock = threading.Lock()
_face_engine: Optional[FaceEngine] = None
_live_lock = threading.Lock()
_live: Optional[LiveSession] = None


def face_engine() -> Optional[FaceEngine]:
    """One shared face engine (models load once, ~1 s)."""
    global _face_engine
    with _face_lock:
        if _face_engine is None and models_available(MODELS_DIR):
            _face_engine = FaceEngine(MODELS_DIR, det_score=0.7, min_face_px=24)
        return _face_engine


def _require_faces() -> FaceEngine:
    eng = face_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Face models are not installed. In the backend folder run: "
                                                    "python download_models.py  then restart the server.")
    return eng


# ===========================================================================
# people
# ===========================================================================
def _read_upload(f: Optional[UploadFile]) -> Optional[bytes]:
    if f is None or not getattr(f, "filename", ""):
        return None
    data = f.file.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=413, detail=f"'{f.filename}' is bigger than 15 MB.")
    return data or None


def _photo_list(front, left, right, extra) -> list:
    photos = []
    for angle, f in (("front", front), ("left", left), ("right", right)):
        b = _read_upload(f)
        if b:
            photos.append((angle, b))
    for f in extra or []:
        b = _read_upload(f)
        if b:
            photos.append(("extra", b))
    return photos


@router.get("/api/people")
def list_people():
    return REGISTRY.list()


@router.post("/api/people")
def register_person(name: str = Form(...), roll_no: str = Form(...), role: str = Form("Student"),
                    department: str = Form(""),
                    front: Optional[UploadFile] = File(None), left: Optional[UploadFile] = File(None),
                    right: Optional[UploadFile] = File(None), extra: List[UploadFile] = File(default=[])):
    eng = _require_faces()
    try:
        person, notes = REGISTRY.enroll(eng, name, roll_no, role, department,
                                        _photo_list(front, left, right, extra), ICFG.duplicate_thr)
    except EnrollError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"person": person, "notes": notes}


@router.post("/api/people/{pid}/photos")
def add_photos(pid: str, front: Optional[UploadFile] = File(None), left: Optional[UploadFile] = File(None),
               right: Optional[UploadFile] = File(None), extra: List[UploadFile] = File(default=[])):
    eng = _require_faces()
    if REGISTRY.get(pid) is None:
        raise HTTPException(status_code=404, detail="Person not found.")
    try:
        person, notes = REGISTRY.enroll(eng, "", "", photos=_photo_list(front, left, right, extra),
                                        duplicate_thr=ICFG.duplicate_thr, person_id=pid)
    except EnrollError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"person": person, "notes": notes}


class PersonPatch(BaseModel):
    name: Optional[str] = None
    roll_no: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None


@router.patch("/api/people/{pid}")
def edit_person(pid: str, body: PersonPatch):
    if body.roll_no:
        for p in REGISTRY.list():
            if p["id"] != pid and p.get("roll_no", "").lower() == body.roll_no.strip().lower():
                raise HTTPException(status_code=400, detail=f"Roll number already used by {p['name']}.")
    try:
        return REGISTRY.update_info(pid, **(body.model_dump() if hasattr(body, "model_dump") else body.dict()))
    except KeyError:
        raise HTTPException(status_code=404, detail="Person not found.")


@router.delete("/api/people/{pid}")
def delete_person(pid: str):
    if not REGISTRY.delete(pid):
        raise HTTPException(status_code=404, detail="Person not found.")
    return {"deleted": pid}


@router.get("/api/people/{pid}/image/{filename}")
def person_photo(pid: str, filename: str):
    path = REGISTRY.photo_path(pid, filename)
    if not path:
        raise HTTPException(status_code=404, detail="No such photo.")
    return FileResponse(path, media_type="image/jpeg")


# ===========================================================================
# live camera
# ===========================================================================
class LiveStart(BaseModel):
    source: Optional[str] = "0"              # "0" webcam, "rtsp://..." CCTV, "http://...", file path
    video_id: Optional[str] = None           # or: an uploaded video (for testing without a camera)
    camera: str = "Main gate"
    mode: str = "entry"                      # entry | exit | door
    line: Optional[List[float]] = None       # door mode: [x1, y1, x2, y2] in 0..1
    flip: bool = False


def _video_path(video_id: str) -> str:
    import re
    if not re.fullmatch(r"[0-9a-f]{8}", video_id or ""):
        raise HTTPException(status_code=404, detail="Unknown video.")
    folder = os.path.join(UPLOAD_DIR, video_id)
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.startswith("original"):
                return os.path.join(folder, name)
    raise HTTPException(status_code=404, detail="Unknown video.")


def live_state() -> dict:
    with _live_lock:
        return _live.status() if _live is not None else {"state": "off"}


def stop_live() -> None:
    global _live
    with _live_lock:
        s, _live = _live, None
    if s is not None:
        s.stop()


@router.post("/api/live/start")
def live_start(body: LiveStart):
    global _live
    if body.mode not in ("entry", "exit", "door"):
        raise HTTPException(status_code=400, detail="mode must be entry, exit or door.")
    line = None
    if body.mode == "door":
        if not body.line or len(body.line) != 4:
            raise HTTPException(status_code=400, detail="Draw the door line first.")
        line = tuple(min(1.0, max(0.0, float(v))) for v in body.line)
    source = _video_path(body.video_id) if body.video_id else (body.source or "0").strip()
    label = f"uploaded video {body.video_id}" if body.video_id else mask_url(source)

    import importlib.util
    if importlib.util.find_spec("ultralytics") is None:
        raise HTTPException(status_code=503, detail="ultralytics is not installed. Run: pip install -r requirements.txt")
    eng = face_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Face models are not installed. Run: python download_models.py")

    stop_live()
    try:
        engine = LiveEngine(REGISTRY, ATTENDANCE, eng, ICFG, camera=body.camera.strip() or "Camera",
                            mode=body.mode, line=line, flip=body.flip)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    model = DEFAULT_YOLO if os.path.exists(DEFAULT_YOLO) else "yolov8n.pt"
    session = LiveSession(source, engine,
                          lambda: YoloLiveTracker(model, TRACKER_YAML, ICFG.live_conf, 0.7, ICFG.live_imgsz),
                          label=label)
    with _live_lock:
        _live = session
    session.start()
    for _ in range(40):                                   # wait up to ~4 s for an early error
        if session.state != "starting":
            break
        time.sleep(0.1)
    st = session.status()
    if st["state"] == "error":
        raise HTTPException(status_code=400, detail=st["error"])
    return st


@router.post("/api/live/stop")
def live_stop():
    stop_live()
    return {"state": "off"}


@router.get("/api/live/status")
def live_status():
    return live_state()


@router.get("/api/live/stream")
def live_stream():
    with _live_lock:
        session = _live
    if session is None:
        raise HTTPException(status_code=404, detail="The live camera is off.")

    def frames():
        last = -1
        idle = 0
        while True:
            with _live_lock:
                current = _live
            if current is not session or not (session.running or session.state == "finished"):
                break
            seq, jpg = session.latest()
            if jpg is not None and seq != last:
                last, idle = seq, 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            else:
                idle += 1
                if session.state == "finished" and idle > 50:
                    break
                time.sleep(0.04)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.get("/api/live/feed")
def live_feed(limit: int = Query(40, ge=1, le=200)):
    with _live_lock:
        session = _live
    if session is None:
        return []
    return list(session.engine.feed)[:limit]


@router.get("/api/live/probe")
def live_probe(source: Optional[str] = None, video_id: Optional[str] = None):
    """A still picture, used to draw the door line for 'door' mode."""
    frame = None
    with _live_lock:
        session = _live
    if session is not None and session.running:
        frame = session.latest_raw()
    if frame is None:
        path = _video_path(video_id) if video_id else (source or "0")
        frame = grab_one_frame(path)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not get a picture from this camera. Check the IP "
                            "address, user name, password and channel, and that this computer is on the "
                            "same network as the CCTV / DVR.")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ===========================================================================
# attendance + alerts
# ===========================================================================
@router.get("/api/attendance")
def attendance(day: Optional[str] = None):
    return ATTENDANCE.day_report(REGISTRY.list(), day)


@router.get("/api/attendance.csv")
def attendance_csv(day: Optional[str] = None):
    d = valid_day(day)
    return Response(ATTENDANCE.day_csv(REGISTRY.list(), d), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="attendance_{d}.csv"'})


@router.get("/api/attendance/days")
def attendance_days():
    return ATTENDANCE.days()


@router.get("/api/log")
def event_log(day: Optional[str] = None):
    return ATTENDANCE.log(day)


class ManualMark(BaseModel):
    person_id: str
    kind: str = "in"


@router.post("/api/attendance/manual")
def manual_mark(body: ManualMark):
    p = REGISTRY.get(body.person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Person not found.")
    if body.kind not in ("in", "out"):
        raise HTTPException(status_code=400, detail="kind must be in or out.")
    return ATTENDANCE.mark(body.kind, p, camera="manual", manual=True)


@router.get("/api/alerts")
def alerts(day: Optional[str] = None):
    return ATTENDANCE.alerts(day)


class Review(BaseModel):
    reviewed: bool = True


@router.post("/api/alerts/{alert_id}/review")
def review_alert(alert_id: int, body: Review):
    if not ATTENDANCE.review_alert(alert_id, body.reviewed):
        raise HTTPException(status_code=404, detail="Alert not found.")
    return {"id": alert_id, "reviewed": body.reviewed}


@router.get("/api/snapshot/{day}/{filename}")
def snapshot(day: str, filename: str):
    path = ATTENDANCE.snapshot_path(f"{day}/{filename}")
    if not path:
        raise HTTPException(status_code=404, detail="No such picture.")
    return FileResponse(path, media_type="image/jpeg")
