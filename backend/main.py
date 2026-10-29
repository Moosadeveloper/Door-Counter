"""
main.py
-------
FastAPI backend for the AI Restaurant Door Counter (final year project).

  GET    /                          the web app (frontend/index.html)
  GET    /api/health                is YOLO / face model / ffmpeg available?
  POST   /upload                    upload a video -> video_id (+ first frame)
  GET    /frame/{id}                first frame (the user draws the door line on it)
  POST   /analyze/{id}              start analysis (background job)
  GET    /status/{id}               progress of the job
  GET    /results/{id}              results JSON (counts, people, events, timeline)
  GET    /video/{id}                annotated video (H.264)
  GET    /csv/{id}?kind=people|events
  GET    /person/{id}/{person_id}   face thumbnail (or body snapshot if no face)
  DELETE /video/{id}                delete the video and every file made from it

Institution / attendance (see institution_api.py):
  /api/people ...     register people with front / left / right photos
  /api/live ...       start / stop the live camera, MJPEG stream, live feed
  /api/attendance ... daily present / absent register, CSV export
  /api/alerts ...     unknown person / face hidden alerts with photos

No database: everything is plain files under uploads/<video_id>/.
Start:  uvicorn main:app        then open  http://localhost:8000
"""
import importlib.util
import json
import os
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import AnalysisConfig
import institution_api
from faces import models_available
from geometry import DoorLine
from render import find_ffmpeg
from video_processor import MODELS_DIR, probe_video, process_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(title="AI Restaurant Door Counter")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def video_folder(video_id: str) -> str:
    """Validate the id (blocks path tricks like '../') and return its folder."""
    if not re.fullmatch(r"[0-9a-f]{8}", video_id):
        raise HTTPException(status_code=404, detail="Unknown video_id.")
    folder = os.path.join(UPLOAD_DIR, video_id)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail="video_id not found. Upload first.")
    return folder


def find_original(folder: str) -> str:
    for name in os.listdir(folder):
        if name.startswith("original"):
            return os.path.join(folder, name)
    raise HTTPException(status_code=404, detail="Original video file is missing.")


def save_first_frame(video_path: str, folder: str) -> None:
    cap = cv2.VideoCapture(video_path)
    frame = None
    for _ in range(30):                       # skip black fade-in frames
        ok, fr = cap.read()
        if not ok:
            break
        frame = fr
        if fr.mean() > 8:
            break
    cap.release()
    if frame is None:
        raise RuntimeError("no frames")
    cv2.imwrite(os.path.join(folder, "first_frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# background jobs (one at a time: the analysis is CPU/GPU heavy)
# ---------------------------------------------------------------------------
JOBS: dict = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def set_status(video_id: str, persist: bool = True, **fields) -> None:
    with LOCK:
        st = JOBS.setdefault(video_id, {"state": "idle", "progress": 0.0, "stage": "", "message": ""})
        st.update(fields)
        snapshot = dict(st)
    if persist:                               # remember state changes across server restarts
        try:
            with open(os.path.join(UPLOAD_DIR, video_id, "status.json"), "w") as fh:
                json.dump(snapshot, fh)
        except OSError:
            pass


def read_status(video_id: str) -> dict:
    with LOCK:
        st = JOBS.get(video_id)
        if st:
            return dict(st)
    folder = os.path.join(UPLOAD_DIR, video_id)
    if os.path.isfile(os.path.join(folder, "results.json")):
        return {"state": "done", "progress": 1.0, "stage": "Finished", "message": ""}
    path = os.path.join(folder, "status.json")
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                st = json.load(fh)
            if st.get("state") in ("queued", "running"):
                return {"state": "error", "progress": 0.0, "stage": "",
                        "message": "The server restarted during the analysis. Please analyse again."}
            return st
        except (OSError, ValueError):
            pass
    return {"state": "idle", "progress": 0.0, "stage": "", "message": ""}


def run_job(video_id: str, original: str, folder: str, cfg: AnalysisConfig):
    try:
        set_status(video_id, state="running", progress=0.0, stage="Starting", message="")
        summary, _ = process_video(
            input_path=original, output_dir=folder, config=cfg,
            registry=institution_api.REGISTRY, inst_cfg=institution_api.ICFG,
            progress_cb=lambda p, stage: set_status(video_id, persist=False, progress=round(p, 4), stage=stage))
        set_status(video_id, state="done", progress=1.0, stage="Finished")
        return summary
    except Exception as e:                    # report any failure to the UI
        traceback.print_exc()
        set_status(video_id, state="error", message=str(e) or e.__class__.__name__)
        raise


def clear_outputs(folder: str) -> None:
    for name in ("results.json", "results.csv", "events.csv", "processed_video.mp4",
                 "processed_video.mp4.raw.mp4", "status.json"):
        try:
            os.remove(os.path.join(folder, name))
        except OSError:
            pass
    for sub in ("faces", "people"):
        shutil.rmtree(os.path.join(folder, sub), ignore_errors=True)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
app.include_router(institution_api.router)


@app.on_event("shutdown")
def _shutdown():
    institution_api.stop_live()


@app.get("/api/health")
def health():
    return {"status": "ok",
            "yolo_installed": importlib.util.find_spec("ultralytics") is not None,
            "face_models": models_available(MODELS_DIR),
            "ffmpeg": find_ffmpeg() is not None,
            "registered_people": institution_api.REGISTRY.count(),
            "live": institution_api.live_state()}


@app.get("/api/videos")
def list_videos():
    """Uploaded videos - can also be used as a 'camera' in live mode (for testing)."""
    out = []
    for vid in sorted(os.listdir(UPLOAD_DIR)):
        folder = os.path.join(UPLOAD_DIR, vid)
        if not (os.path.isdir(folder) and re.fullmatch(r"[0-9a-f]{8}", vid)):
            continue
        for name in os.listdir(folder):
            if name.startswith("original"):
                out.append({"video_id": vid, "file": name,
                            "modified": os.path.getmtime(os.path.join(folder, name))})
    out.sort(key=lambda v: -v["modified"])
    return out


@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    """Save the uploaded video and return a video_id to use for analysis."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXT))}")
    video_id = uuid.uuid4().hex[:8]
    folder = os.path.join(UPLOAD_DIR, video_id)
    os.makedirs(folder, exist_ok=True)
    saved = os.path.join(folder, f"original{ext}")
    with open(saved, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        info = probe_video(saved)
        save_first_frame(saved, folder)
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not read this video: {e}")
    duration = (info["frames"] / info["fps"]) if info["frames"] else None
    return {"video_id": video_id, "filename": file.filename, "width": info["width"],
            "height": info["height"], "fps": round(info["fps"], 2), "frames": info["frames"],
            "duration_seconds": None if duration is None else round(duration, 1)}


@app.get("/frame/{video_id}")
def first_frame(video_id: str):
    path = os.path.join(video_folder(video_id), "first_frame.jpg")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No preview frame.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/analyze/{video_id}")
def analyze_video(
    video_id: str,
    line_position: float = Query(0.5, ge=0, le=1, description="legacy: vertical line position"),
    x1: Optional[float] = Query(None, ge=0, le=1), y1: Optional[float] = Query(None, ge=0, le=1),
    x2: Optional[float] = Query(None, ge=0, le=1), y2: Optional[float] = Query(None, ge=0, le=1),
    flip: bool = False,
    stride: int = Query(1, ge=1, le=10),
    use_faces: bool = True,
    imgsz: int = Query(768, ge=320, le=1920),
    capacity: Optional[int] = Query(None, ge=0, le=1000, description="alert if more people than this are inside"),
    wait: bool = Query(False, description="true = old behaviour: block until finished and return the summary"),
):
    """Start the analysis. Door line = (x1,y1)-(x2,y2) in 0..1 image coordinates."""
    folder = video_folder(video_id)
    original = find_original(folder)

    given = [v is not None for v in (x1, y1, x2, y2)]
    if any(given) and not all(given):
        raise HTTPException(status_code=400, detail="Give all of x1, y1, x2, y2 (or none of them).")
    line = (x1, y1, x2, y2) if all(given) else (line_position, 0.0, line_position, 1.0)
    cfg = AnalysisConfig(line=line, flip=flip, stride=stride, use_faces=use_faces, imgsz=imgsz,
                         capacity=capacity).normalise()

    try:
        info = probe_video(original)
        DoorLine(cfg.line, info["width"], info["height"], cfg.flip)        # validates the line
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if read_status(video_id)["state"] in ("queued", "running"):
        raise HTTPException(status_code=409, detail="This video is already being analysed.")

    clear_outputs(folder)
    set_status(video_id, state="queued", progress=0.0, stage="Waiting for the analyser", message="")

    if wait:
        try:
            return run_job(video_id, original, folder, cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    EXECUTOR.submit(_safe_run, video_id, original, folder, cfg)
    return {"video_id": video_id, "state": "queued"}


def _safe_run(video_id, original, folder, cfg):
    try:
        run_job(video_id, original, folder, cfg)
    except Exception:
        pass                                   # already stored in the job status


@app.get("/status/{video_id}")
def get_status(video_id: str):
    video_folder(video_id)
    return read_status(video_id)


@app.get("/results/{video_id}")
def get_results(video_id: str):
    path = os.path.join(video_folder(video_id), "results.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No results yet. Run /analyze first.")
    with open(path, encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


@app.get("/video/{video_id}")
def get_processed_video(video_id: str):
    path = os.path.join(video_folder(video_id), "processed_video.mp4")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No processed video yet. Run /analyze first.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/csv/{video_id}")
def get_csv(video_id: str, kind: str = Query("people", pattern="^(people|events)$")):
    name = "results.csv" if kind == "people" else "events.csv"
    path = os.path.join(video_folder(video_id), name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No results yet. Run /analyze first.")
    return FileResponse(path, media_type="text/csv", filename=name)


@app.get("/person/{video_id}/{person_id}")
def person_image(video_id: str, person_id: int, kind: str = Query("auto", pattern="^(auto|face|body)$")):
    folder = video_folder(video_id)
    face = os.path.join(folder, "faces", f"person_{person_id}.jpg")
    body = os.path.join(folder, "people", f"person_{person_id}.jpg")
    order = {"face": [face], "body": [body]}.get(kind, [face, body])
    for path in order:
        if os.path.isfile(path):
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No image for this person.")


@app.delete("/video/{video_id}")
def delete_video(video_id: str):
    folder = video_folder(video_id)
    if read_status(video_id)["state"] in ("queued", "running"):
        raise HTTPException(status_code=409, detail="Wait for the analysis to finish first.")
    shutil.rmtree(folder, ignore_errors=True)
    with LOCK:
        JOBS.pop(video_id, None)
    return {"deleted": video_id}


# ---------------------------------------------------------------------------
# frontend (plain HTML/CSS/JS - no Node.js or build step needed)
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/")
    def root():
        return {"status": "ok", "message": "AI Restaurant Door Counter API (frontend folder not found)"}
