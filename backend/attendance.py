"""
attendance.py
-------------
Attendance register + alert log, stored in ONE SQLite file
(data/attendance.db - SQLite is built into Python, nothing to install).

Tables
  attendance  one row per (day, person): first time in, last time out, ...
  log         every IN / OUT event (who, when, which camera, how sure)
  alerts      UNKNOWN person / FACE HIDDEN person, with a snapshot photo

Snapshot photos are saved as files under data/snapshots/<date>/.
"""
import csv
import io
import os
import re
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance (
    day TEXT NOT NULL, person_id TEXT NOT NULL, name TEXT, roll_no TEXT,
    check_in TEXT, check_out TEXT, last_seen TEXT,
    entries INTEGER DEFAULT 0, exits INTEGER DEFAULT 0,
    camera TEXT, score REAL, snapshot TEXT, manual INTEGER DEFAULT 0,
    PRIMARY KEY (day, person_id));
CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL,
    person_id TEXT, name TEXT, roll_no TEXT, kind TEXT NOT NULL,
    camera TEXT, score REAL, snapshot TEXT, manual INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL,
    kind TEXT NOT NULL, camera TEXT, snapshot TEXT, face_snapshot TEXT,
    note TEXT, resolved_name TEXT, reviewed INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS log_day ON log(day);
CREATE INDEX IF NOT EXISTS alerts_day ON alerts(day);
"""

ALERT_TEXT = {
    "unknown": "Unknown person - face is visible but does not match anyone registered",
    "face_hidden": "Face hidden - person entered without showing their face",
}


def _day(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def _hms(ts: datetime) -> str:
    return ts.strftime("%H:%M:%S")


def valid_day(day: Optional[str]) -> str:
    if day and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return day
    return _day(datetime.now())


class AttendanceDB:
    def __init__(self, db_path: str, snap_dir: str, log_cooldown: float = 60.0,
                 snap_max_px: int = 320, snap_quality: int = 60, snap_known: str = "first",
                 snap_keep_days: int = 15):
        self.db_path, self.snap_dir = db_path, snap_dir
        self.log_cooldown = log_cooldown
        self.snap_max_px, self.snap_quality = snap_max_px, snap_quality
        self.snap_known, self.snap_keep_days = snap_known, snap_keep_days
        self._last_cleanup: Optional[str] = None
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(snap_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(SCHEMA)
        self._con.commit()
        self._last_log: Dict[tuple, datetime] = {}

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # ------------------------------------------------------------ snapshots
    def save_snapshot(self, jpg: Optional[bytes], ts: datetime, prefix: str) -> Optional[str]:
        """Returns a relative name like '2026-09-26/in_ab12cd34.jpg'."""
        if not jpg:
            return None
        jpg = self._shrink(jpg)
        self.cleanup_old()
        day = _day(ts)
        folder = os.path.join(self.snap_dir, day)
        os.makedirs(folder, exist_ok=True)
        name = f"{prefix}_{ts.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(jpg)
        return f"{day}/{name}"

    def _shrink(self, jpg: bytes) -> bytes:
        """Storage saver: photo ko chhota karke kam quality par dobara JPEG banao.
        ~100-300 KB ki body photo -> ~10-25 KB."""
        try:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return jpg
            h, w = img.shape[:2]
            k = self.snap_max_px / max(h, w)
            if k < 1:
                img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.snap_quality,
                                                 cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            return buf.tobytes() if ok and len(buf) < len(jpg) else jpg
        except Exception:
            return jpg

    def cleanup_old(self, force: bool = False) -> int:
        """snap_keep_days se purane snapshot folders delete. Din me ek dafa chalta hai.
        Attendance records (database) delete NAHI hote, sirf photos."""
        if self.snap_keep_days <= 0:
            return 0
        today = _day(datetime.now())
        if not force and self._last_cleanup == today:
            return 0
        self._last_cleanup = today
        cutoff = _day(datetime.now() - timedelta(days=self.snap_keep_days))
        removed = 0
        for name in os.listdir(self.snap_dir):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name) and name < cutoff:
                shutil.rmtree(os.path.join(self.snap_dir, name), ignore_errors=True)
                removed += 1
        return removed

    def snapshot_path(self, rel: str) -> Optional[str]:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}/[A-Za-z0-9_]+\.jpg", rel or ""):
            return None
        path = os.path.join(self.snap_dir, rel)
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------ in / out
    def mark(self, kind: str, person: dict, ts: Optional[datetime] = None, camera: str = "",
             score: Optional[float] = None, snapshot_jpg: Optional[bytes] = None,
             manual: bool = False) -> Optional[dict]:
        """kind = 'in' or 'out'. Returns the logged event, or None if it was a
        repeat of the same event within the cooldown (camera sees the same
        student many times per second - log it once)."""
        if kind not in ("in", "out"):
            raise ValueError(kind)
        ts = ts or datetime.now()
        pid = person["id"]
        key = (pid, kind, camera)
        last = self._last_log.get(key)
        if not manual and last is not None and (ts - last).total_seconds() < self.log_cooldown:
            with self._lock:                                  # just refresh "last seen"
                self._con.execute("UPDATE attendance SET last_seen=? WHERE day=? AND person_id=?",
                                  (_hms(ts), _day(ts), pid))
                self._con.commit()
            return None
        self._last_log[key] = ts
        if self.snap_known == "none":
            snapshot_jpg = None
        elif self.snap_known == "first" and snapshot_jpg is not None:
            with self._lock:                                  # aaj ki photo pehle se hai? dobara mat save karo
                had = self._con.execute("SELECT snapshot FROM attendance WHERE day=? AND person_id=?",
                                        (_day(ts), pid)).fetchone()
            if had is not None and had["snapshot"]:
                snapshot_jpg = None
        snap = self.save_snapshot(snapshot_jpg, ts, kind)
        day, t = _day(ts), _hms(ts)

        with self._lock:
            row = self._con.execute("SELECT * FROM attendance WHERE day=? AND person_id=?",
                                    (day, pid)).fetchone()
            if row is None:
                self._con.execute(
                    "INSERT INTO attendance (day, person_id, name, roll_no, check_in, check_out, last_seen,"
                    " entries, exits, camera, score, snapshot, manual) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (day, pid, person.get("name"), person.get("roll_no"),
                     t if kind == "in" else None, t if kind == "out" else None, t,
                     1 if kind == "in" else 0, 1 if kind == "out" else 0,
                     camera, score, snap, int(manual)))
            elif kind == "in":
                self._con.execute(
                    "UPDATE attendance SET check_in=COALESCE(check_in, ?), check_out=NULL, last_seen=?,"
                    " entries=entries+1, camera=?, score=MAX(COALESCE(score,0), ?),"
                    " snapshot=COALESCE(snapshot, ?), manual=MAX(manual, ?) WHERE day=? AND person_id=?",
                    (t, t, camera, score or 0, snap, int(manual), day, pid))
            else:
                self._con.execute(
                    "UPDATE attendance SET check_out=?, last_seen=?, exits=exits+1,"
                    " manual=MAX(manual, ?) WHERE day=? AND person_id=?",
                    (t, t, int(manual), day, pid))
            cur = self._con.execute(
                "INSERT INTO log (ts, day, person_id, name, roll_no, kind, camera, score, snapshot, manual)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts.isoformat(timespec="seconds"), day, pid, person.get("name"), person.get("roll_no"),
                 kind, camera, score, snap, int(manual)))
            self._con.commit()
            return {"id": cur.lastrowid, "time": t, "day": day, "person_id": pid,
                    "name": person.get("name"), "roll_no": person.get("roll_no"), "kind": kind,
                    "camera": camera, "score": None if score is None else round(float(score), 3),
                    "snapshot": snap, "manual": manual}

    # ------------------------------------------------------------ alerts
    def add_alert(self, kind: str, ts: Optional[datetime] = None, camera: str = "",
                  body_jpg: Optional[bytes] = None, face_jpg: Optional[bytes] = None,
                  note: str = "") -> dict:
        ts = ts or datetime.now()
        snap = self.save_snapshot(body_jpg, ts, kind)
        face = self.save_snapshot(face_jpg, ts, kind + "_face")
        note = note or ALERT_TEXT.get(kind, kind)
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO alerts (ts, day, kind, camera, snapshot, face_snapshot, note)"
                " VALUES (?,?,?,?,?,?,?)",
                (ts.isoformat(timespec="seconds"), _day(ts), kind, camera, snap, face, note))
            self._con.commit()
        return {"id": cur.lastrowid, "time": _hms(ts), "day": _day(ts), "kind": kind, "camera": camera,
                "snapshot": snap, "face_snapshot": face, "note": note, "resolved_name": None,
                "reviewed": False}

    def resolve_alert(self, alert_id: int, name: str) -> None:
        """The camera later DID recognise the person behind an alert."""
        with self._lock:
            self._con.execute("UPDATE alerts SET resolved_name=? WHERE id=?", (name, alert_id))
            self._con.commit()

    def review_alert(self, alert_id: int, reviewed: bool = True) -> bool:
        with self._lock:
            cur = self._con.execute("UPDATE alerts SET reviewed=? WHERE id=?", (int(reviewed), alert_id))
            self._con.commit()
            return cur.rowcount > 0

    def alerts(self, day: Optional[str] = None, limit: int = 500) -> List[dict]:
        with self._lock:
            rows = self._con.execute("SELECT * FROM alerts WHERE day=? ORDER BY id DESC LIMIT ?",
                                     (valid_day(day), limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["time"] = d["ts"][11:19]
            d["reviewed"] = bool(d["reviewed"])
            out.append(d)
        return out

    # ------------------------------------------------------------ reports
    def log(self, day: Optional[str] = None, limit: int = 1000) -> List[dict]:
        with self._lock:
            rows = self._con.execute("SELECT * FROM log WHERE day=? ORDER BY id DESC LIMIT ?",
                                     (valid_day(day), limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["time"] = d["ts"][11:19]
            d["manual"] = bool(d["manual"])
            out.append(d)
        return out

    def days(self) -> List[str]:
        with self._lock:
            rows = self._con.execute(
                "SELECT day FROM attendance UNION SELECT day FROM alerts ORDER BY day DESC").fetchall()
        return [r[0] for r in rows]

    def day_report(self, people: List[dict], day: Optional[str] = None) -> dict:
        """Every REGISTERED person with Present / Absent for one day."""
        day = valid_day(day)
        with self._lock:
            rows = {r["person_id"]: dict(r) for r in self._con.execute(
                "SELECT * FROM attendance WHERE day=?", (day,)).fetchall()}
            n_alerts = self._con.execute("SELECT COUNT(*) FROM alerts WHERE day=?", (day,)).fetchone()[0]
        out, present, inside = [], 0, 0
        known = set()
        for p in people:
            known.add(p["id"])
            r = rows.get(p["id"])
            rec = {"person_id": p["id"], "name": p["name"], "roll_no": p.get("roll_no", ""),
                   "role": p.get("role", ""), "department": p.get("department", ""),
                   "status": "absent", "check_in": None, "check_out": None, "last_seen": None,
                   "entries": 0, "exits": 0, "camera": None, "score": None, "snapshot": None,
                   "manual": False, "inside": False, "time_inside": None}
            if r is not None and (r["check_in"] or r["check_out"]):
                present += 1
                rec.update({k: r[k] for k in ("check_in", "check_out", "last_seen", "entries", "exits",
                                              "camera", "snapshot")})
                rec["score"] = None if r["score"] is None else round(float(r["score"]), 3)
                rec["manual"] = bool(r["manual"])
                rec["status"] = "present"
                rec["inside"] = bool(r["check_in"]) and not r["check_out"]
                inside += rec["inside"]
                if r["check_in"] and r["check_out"]:
                    a = datetime.strptime(r["check_in"], "%H:%M:%S")
                    b = datetime.strptime(r["check_out"], "%H:%M:%S")
                    secs = max(0, int((b - a).total_seconds()))
                    rec["time_inside"] = f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}"
            out.append(rec)
        # people who were deleted from the registry but have attendance that day
        for pid, r in rows.items():
            if pid not in known:
                out.append({"person_id": pid, "name": (r["name"] or "?") + " (deleted)",
                            "roll_no": r["roll_no"] or "", "role": "", "department": "",
                            "status": "present", "check_in": r["check_in"], "check_out": r["check_out"],
                            "last_seen": r["last_seen"], "entries": r["entries"], "exits": r["exits"],
                            "camera": r["camera"], "score": r["score"], "snapshot": r["snapshot"],
                            "manual": bool(r["manual"]), "inside": False, "time_inside": None})
        out.sort(key=lambda x: (x["status"] != "present", x["check_in"] or "99", x["name"].lower()))
        return {"day": day, "total": len(people), "present": present,
                "absent": max(0, len(people) - sum(1 for p in people if rows.get(p["id"]))),
                "inside_now": inside, "alerts": n_alerts, "people": out}

    def day_csv(self, people: List[dict], day: Optional[str] = None) -> str:
        rep = self.day_report(people, day)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "roll_no", "name", "role", "department", "status", "check_in", "check_out",
                    "time_inside", "entries", "exits", "camera", "manual"])
        for p in rep["people"]:
            w.writerow([rep["day"], p["roll_no"], p["name"], p["role"], p["department"],
                        p["status"].upper(), p["check_in"] or "", p["check_out"] or "",
                        p["time_inside"] or "", p["entries"], p["exits"], p["camera"] or "",
                        "yes" if p["manual"] else ""])
        return buf.getvalue()
