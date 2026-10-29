"""
render.py
---------
Second pass over the video: draws the door line, person boxes with their
FINAL person IDs, live ENTERED / EXITED / INSIDE counters and short
"P3 ENTERED" pop-ups, then converts the file to H.264 so that it plays in
every browser (OpenCV's own "mp4v" output does NOT play in Chrome/Edge).
"""
import os
import shutil
import subprocess
from bisect import bisect_right
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from events import PersonTimeline, format_time

GRAY = (200, 200, 200)       # BGR colours
GREEN = (80, 200, 60)
ORANGE = (0, 140, 255)
YELLOW = (0, 230, 255)


def find_ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg                    # pip install imageio-ffmpeg (bundles ffmpeg, free)
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def transcode_h264(src: str, dst: str) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-i", src,
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60 * 60)
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------
def _text(img, s, org, scale, color, thick, bg=None):
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return w, h


def _dashed_line(img, p1, p2, color, thick, dash=18):
    x1, y1 = p1
    x2, y2 = p2
    length = float(np.hypot(x2 - x1, y2 - y1))
    n = max(1, int(length // dash))
    for i in range(0, n, 2):
        a, b = i / n, min(1.0, (i + 1) / n)
        cv2.line(img, (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
                 (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)), color, thick, cv2.LINE_AA)


def _corner_box(img, x1, y1, x2, y2, color, thick):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thick - 1), cv2.LINE_AA)
    L = max(10, int(0.2 * min(x2 - x1, y2 - y1)))
    for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * L, cy), color, thick + 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * L), color, thick + 1, cv2.LINE_AA)


def _draw_door(img, door, s):
    p1 = (int(door.p1[0]), int(door.p1[1]))
    p2 = (int(door.p2[0]), int(door.p2[1]))
    _dashed_line(img, p1, p2, YELLOW, max(2, int(3 * s)))
    # arrow from the middle of the line toward the INSIDE
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    nx, ny = door.inside_normal()
    L = int(70 * s)
    tip = (int(mx + nx * L), int(my + ny * L))
    cv2.arrowedLine(img, (mx, my), tip, YELLOW, max(2, int(3 * s)), cv2.LINE_AA, tipLength=0.3)
    _text(img, "IN", (tip[0] + 6, tip[1] + 6), 0.6 * s, YELLOW, max(1, int(2 * s)), bg=(0, 0, 0))


# ---------------------------------------------------------------------------
# main renderer
# ---------------------------------------------------------------------------
def render_video(input_path: str, out_path: str, fps: float, size: Tuple[int, int],
                 tracklets: dict, person_of_track: Dict[int, int],
                 timelines: List[PersonTimeline], persons: dict, door,
                 progress_cb: Optional[Callable[[float], None]] = None,
                 total_frames: int = 0, stride: int = 1,
                 capacity: Optional[int] = None) -> Tuple[bool, str]:
    """Returns (browser_playable, final_path)."""
    width, height = size
    s = max(0.6, min(1.6, height / 720.0))
    thick = max(1, int(round(2 * s)))

    # ---- per-frame boxes -------------------------------------------------
    by_frame: Dict[int, List[tuple]] = {}
    for tid, tl in tracklets.items():
        pid = person_of_track.get(tid)
        if pid is None:
            continue                                   # dropped as noise
        for f, b in zip(tl.frames, tl.boxes):
            by_frame.setdefault(f, []).append((pid, b))

    # ---- per-person state lookup ----------------------------------------
    tl_of = {t.person_id: t for t in timelines}
    ev_frames = {p: [e.frame for e in t.events] for p, t in tl_of.items()}
    ev_kinds = {p: [e.kind for e in t.events] for p, t in tl_of.items()}
    v_starts = {p: [v.start_frame for v in t.visits] for p, t in tl_of.items()}

    def status(pid: int, f: int) -> str:
        t = tl_of.get(pid)
        if t is None:
            return "outside"
        i = bisect_right(ev_frames[pid], f)
        if i == 0:
            return "inside" if t.initial_side == 1 else "outside"
        return "inside" if ev_kinds[pid][i - 1] == "entry" else "left"

    # ---- global counters --------------------------------------------------
    all_events = sorted(((e.frame, p, e.kind) for p, t in tl_of.items() for e in t.events))
    entry_frames = sorted(fr for fr, _, k in all_events if k == "entry")
    exit_frames = sorted(fr for fr, _, k in all_events if k == "exit")
    ev_frame_list = [fr for fr, _, _ in all_events]
    changes = []
    for t in timelines:
        for v in t.visits:
            changes.append((v.start_frame, 1))
            if v.end_frame is not None:
                changes.append((v.end_frame, -1))
    changes.sort()
    ch_frames = [c[0] for c in changes]
    ch_cum = list(np.cumsum([c[1] for c in changes])) if changes else []

    raw_path = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create the output video file.")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError("Could not re-open the input video for rendering.")

    current: List[tuple] = []
    last_hit = -10 ** 9          # last frame that had boxes
    f = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f in by_frame:                       # detections for this frame
                current, last_hit = by_frame[f], f
            elif f - last_hit >= stride:            # processed frame with nobody in it
                current = []                        # (frames in between just hold the last boxes)

            _draw_door(frame, door, s)

            for pid, (x1, y1, x2, y2) in current:
                st = status(pid, f)
                color = {"inside": GREEN, "left": ORANGE}.get(st, GRAY)
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                _corner_box(frame, x1i, y1i, x2i, y2i, color, thick)

                ident = persons.get(pid)
                tag = f"P{pid}" + (" face" if ident is not None and ident.face_samples else "")
                tag_bg = (0, 0, 0)
                if ident is not None and getattr(ident, "name", None):
                    tag, tag_bg = f"{ident.name} ({ident.roll_no})", (40, 120, 30)
                elif ident is not None and getattr(ident, "flag", None) == "unknown":
                    tag, tag_bg = f"P{pid} UNKNOWN", (30, 30, 200)
                elif ident is not None and getattr(ident, "flag", None) == "face_hidden":
                    tag, tag_bg = f"P{pid} FACE HIDDEN", (0, 110, 220)
                _text(frame, tag, (x1i, max(int(18 * s), y1i - 6)), 0.6 * s, (255, 255, 255), thick, bg=tag_bg)

                if st == "inside":
                    i = bisect_right(v_starts.get(pid, []), f)
                    if i:
                        secs = (f - v_starts[pid][i - 1]) / fps
                        _text(frame, format_time(secs), (x1i, min(height - 6, y2i + int(18 * s))),
                              0.5 * s, (255, 255, 255), max(1, thick - 1), bg=(0, 0, 0))

            # ---- counters panel --------------------------------------------
            ent = bisect_right(entry_frames, f)
            ext = bisect_right(exit_frames, f)
            k = bisect_right(ch_frames, f)
            inside = int(ch_cum[k - 1]) if k else 0
            pw, ph = int(250 * s), int(112 * s)
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            _text(frame, f"ENTERED  {ent}", (22, int(10 + 30 * s)), 0.65 * s, GREEN, thick)
            _text(frame, f"EXITED   {ext}", (22, int(10 + 62 * s)), 0.65 * s, ORANGE, thick)
            _text(frame, f"INSIDE   {inside}", (22, int(10 + 94 * s)), 0.65 * s, (255, 255, 255), thick)

            if capacity is not None and inside > capacity:
                msg = f"OVER CAPACITY  {inside} / {capacity}"
                (tw, th_), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, thick + 1)
                bx = int(width / 2 - tw / 2) - 14
                flash = 200 if (f // max(1, int(fps * 0.4))) % 2 == 0 else 90     # blink
                cv2.rectangle(frame, (bx, 6), (bx + tw + 28, int(14 + th_ + 14)), (0, 0, flash), -1)
                _text(frame, msg, (bx + 14, int(10 + th_ + 6)), 0.75 * s, (255, 255, 255), thick + 1)

            # ---- recent event pop-ups ----------------------------------------
            hi = bisect_right(ev_frame_list, f)
            lo = bisect_right(ev_frame_list, f - int(2.5 * fps))
            for n, idx in enumerate(range(max(lo, hi - 3), hi)):
                fr, pid, kind = all_events[idx]
                who = getattr(persons.get(pid), "name", None) or f"P{pid}"
                label = f"{who} {'ENTERED' if kind == 'entry' else 'EXITED'}  {format_time(fr / fps)}"
                col = GREEN if kind == "entry" else ORANGE
                _text(frame, label, (int(width / 2 - 110 * s), int(30 * s + n * 30 * s)),
                      0.65 * s, col, thick, bg=(0, 0, 0))

            _text(frame, format_time(f / fps), (12, height - 12), 0.55 * s, (230, 230, 230), max(1, thick - 1))

            writer.write(frame)
            f += 1
            if progress_cb and total_frames and f % 25 == 0:
                progress_cb(min(1.0, f / total_frames))
    finally:
        cap.release()
        writer.release()

    if transcode_h264(raw_path, out_path):
        os.remove(raw_path)
        return True, out_path
    os.replace(raw_path, out_path)          # fallback: still downloadable, may not play in browser
    return False, out_path
