"""
Tests for the institution part: registration, attendance register, live
recognition + UNKNOWN / FACE HIDDEN alerts, door-line in/out, and names in
the offline video analysis. Faces are faked (no models / camera needed).

Run:  python -m tests.test_institution
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attendance import AttendanceDB                         # noqa: E402
from config import AnalysisConfig, InstitutionConfig        # noqa: E402
from detectors import Detection                             # noqa: E402
from faces import FaceSample                                # noqa: E402
from live import LiveEngine                                 # noqa: E402
from registry import EnrollError, PersonRegistry            # noqa: E402

RNG = np.random.RandomState(7)
BASE = {n: (lambda v: v / np.linalg.norm(v))(RNG.randn(128).astype(np.float32)) for n in
        ("ali", "sara", "bilal", "stranger")}


def _rs(*key):
    import zlib
    return np.random.RandomState(zlib.crc32(repr(key).encode()))


def angled(name, angle, noise=0.18, seed=0):
    """A fake fingerprint: base face + a fixed offset per head angle (norm 1.0,
    so a side view really differs from the front) + random noise (norm `noise`)."""
    off = _rs(name, angle).randn(128).astype(np.float32)
    off = off / np.linalg.norm(off) * (0.0 if angle == "front" else 1.0)
    nz = _rs(name, angle, seed).randn(128).astype(np.float32)
    v = BASE[name] + off + noise * nz / np.linalg.norm(nz)
    return v / np.linalg.norm(v)


def photo(name, angle="front"):
    """Registration photo: the name + angle are hidden in the pixels for the fake engine."""
    img = np.zeros((200, 200, 3), np.uint8)
    cv2.putText(img, f"{name}|{angle}", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".png", img)
    FakeFaces.photos[buf.tobytes()[-64:]] = (name, angle)
    return buf.tobytes()


class FakeFaces:
    photos = {}

    def __init__(self):
        self.live_faces = {}           # track_id -> (name or None for hidden, angle)

    def extract_largest(self, img):
        ok, buf = cv2.imencode(".png", img)
        key = buf.tobytes()[-64:]
        name, angle = self.photos.get(key, (None, None))
        if name is None or name == "noface":
            return None
        return FaceSample(angled(name, angle, 0.12), 0.9, b"jpgface", np.zeros((112, 112, 3), np.uint8))

    def embed_aligned(self, aligned):
        return None

    def extract(self, frame, box, meta=None):
        who = (meta or {}).get("who")
        if who is None:                # face hidden / turned away
            return None
        seed = int((meta or {}).get("n", 0))
        return FaceSample(angled(who, (meta or {}).get("angle", "front"), 0.25, seed), 0.8, b"jpgface")


class Env:
    def __init__(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = PersonRegistry(os.path.join(self.tmp, "reg"))
        self.att = AttendanceDB(os.path.join(self.tmp, "a.db"), os.path.join(self.tmp, "snaps"), 60)
        self.faces = FakeFaces()

    def enroll(self, name, roll, angles=("front", "left", "right")):
        return self.reg.enroll(self.faces, name.title(), roll, photos=[(a, photo(name, a)) for a in angles])

    def close(self):
        self.att.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def feed(engine, script, start, frames, fps=10):
    """script(f) -> list of (track_id, box, meta). Runs the engine frame by frame."""
    frame = np.full((480, 640, 3), 80, np.uint8)
    for f in range(frames):
        dets = [Detection(tid, box, 0.9, dict(meta, n=f)) for tid, box, meta in script(f)]
        engine.process(frame.copy(), dets, start + timedelta(seconds=f / fps))


# ---------------------------------------------------------------------------
def test_enroll_and_match_from_any_angle():
    e = Env()
    try:
        p, notes = e.enroll("ali", "IU-001")
        assert p["angles"] == ["front", "left", "right"] and p["samples"] == 3, p
        e.enroll("sara", "IU-002")
        cfg = InstitutionConfig()
        for angle in ("front", "left", "right"):
            m = e.reg.match(angled("ali", angle, 0.2, seed=5), cfg.recog_thr, cfg.recog_margin)
            assert m is not None and m.name == "Ali", (angle, m)
        assert e.reg.match(BASE["stranger"], cfg.recog_thr, cfg.recog_margin) is None
        # survives a restart (loaded from disk)
        again = PersonRegistry(e.reg.root)
        assert again.count() == 2 and again.match(angled("sara", "left", 0.2, 3), 0.4) is not None
    finally:
        e.close()


def test_enroll_rejects_bad_input():
    e = Env()
    try:
        e.enroll("ali", "IU-001")
        for kwargs, words in [
            (dict(name="Ali 2", roll="IU-001", who="bilal"), "already registered"),   # same roll no
            (dict(name="Ali copy", roll="IU-009", who="ali"), "looks like"),           # same face twice
        ]:
            try:
                e.reg.enroll(e.faces, kwargs["name"], kwargs["roll"], photos=[("front", photo(kwargs["who"]))])
                raise AssertionError("should have failed")
            except EnrollError as err:
                assert words in str(err), err
        try:
            e.reg.enroll(e.faces, "Nobody", "IU-010", photos=[("front", photo("noface"))])
            raise AssertionError("should have failed")
        except EnrollError as err:
            assert "No usable face" in str(err)
        _, notes = e.enroll("bilal", "IU-003", angles=("front",))
        assert any("left and right" in n for n in notes), notes
    finally:
        e.close()


def test_entry_camera_marks_present_and_alerts():
    e = Env()
    try:
        e.enroll("ali", "IU-001")
        e.enroll("sara", "IU-002")
        eng = LiveEngine(e.reg, e.att, e.faces, InstitutionConfig(), camera="Gate 1", mode="entry")
        start = datetime(2026, 9, 26, 8, 0, 0)

        def script(f):
            out = []
            grow = min(120, f * 3)
            if f < 60:        # Ali walks toward the camera, side face then front
                out.append((1, (100, 150 - grow, 180, 330), {"who": "ali", "angle": "left" if f < 20 else "front"}))
            if f < 80:        # a stranger with a visible face
                out.append((2, (300, 150 - grow, 380, 330), {"who": "stranger"}))
            if 10 <= f < 90:  # someone with a covered face walking in
                out.append((3, (450, 200 - grow, 530, 360), {"who": None}))
            return out
        feed(eng, script, start, 100)

        rep = e.att.day_report(e.reg.list(), "2026-09-26")
        ali = next(p for p in rep["people"] if p["roll_no"] == "IU-001")
        sara = next(p for p in rep["people"] if p["roll_no"] == "IU-002")
        assert ali["status"] == "present" and ali["check_in"].startswith("08:00:0"), ali
        assert sara["status"] == "absent"
        assert rep["present"] == 1 and rep["absent"] == 1
        kinds = sorted(a["kind"] for a in e.att.alerts("2026-09-26"))
        assert kinds == ["face_hidden", "unknown"], kinds
        assert all(a["snapshot"] for a in e.att.alerts("2026-09-26"))
        assert len(e.att.log("2026-09-26")) == 1                 # Ali logged once, not every frame
        assert e.att.day_csv(e.reg.list(), "2026-09-26").count("PRESENT") == 1
    finally:
        e.close()


def test_door_mode_in_and_out_with_late_recognition():
    e = Env()
    try:
        e.enroll("ali", "IU-001")
        # vertical door line at x=320, inside = left side
        eng = LiveEngine(e.reg, e.att, e.faces, InstitutionConfig(), mode="door", line=(0.5, 0, 0.5, 1))
        start = datetime(2026, 9, 26, 9, 0, 0)

        def script(f):
            if f < 40:                                   # walks right -> left (IN), face visible only late
                x = 560 - f * 10
                meta = {"who": "ali" if f >= 25 else None}
            elif f < 100:
                return []                                # inside, out of view
            elif f < 140:                                # walks back out (new tracker ID)
                x = 160 + (f - 100) * 10
                meta = {"who": "ali"}
            else:
                return []
            tid = 1 if f < 40 else 2
            return [(tid, (x - 40, 150, x + 40, 400), meta)]
        feed(eng, script, start, 150)

        log = sorted(e.att.log("2026-09-26"), key=lambda r: r["id"])
        assert [r["kind"] for r in log] == ["in", "out"], log
        assert log[0]["time"] < "09:00:03", log[0]["time"]        # crossing time, not recognition time
        row = next(p for p in e.att.day_report(e.reg.list(), "2026-09-26")["people"])
        assert row["check_in"] and row["check_out"] and row["time_inside"], row
    finally:
        e.close()


def test_video_analysis_uses_registered_names():
    from tests.fakes import Actor, FakeDetector, Scene, write_scene_video
    from video_processor import process_video

    e = Env()
    try:
        e.enroll("ali", "IU-001")

        class NamedFaces(FakeFaces):
            def extract(self, frame, box, meta=None):
                if not meta or not meta.get("face"):
                    return None
                who = {"A": "ali", "B": "stranger"}[meta["gt"]]
                return FaceSample(angled(who, "front", 0.3, int(box[0] * 7 + box[1])), 0.9, b"x")

        A = Actor("A", (40, 40, 220), [(0, 560, 300), (80, 150, 300), (199, 150, 300)], [(0, 199, 1)])
        B = Actor("B", (220, 60, 60), [(20, 600, 280), (110, 120, 280), (199, 120, 280)], [(20, 199, 2)])
        C = Actor("C", (60, 200, 60), [(0, 560, 250), (90, 140, 250), (199, 140, 250)], [(0, 199, 3)],
                  face_frames=[])
        scene = Scene([A, B, C], frames=200)
        src = os.path.join(e.tmp, "in.mp4")
        write_scene_video(src, scene)
        s, _ = process_video(src, os.path.join(e.tmp, "out"), config=AnalysisConfig(),
                             detector=FakeDetector(scene), face_engine=NamedFaces(), registry=e.reg)
        by_track = {r["track_ids"][0]: r for r in s["records"]}
        assert by_track[1]["name"] == "Ali" and by_track[1]["roll_no"] == "IU-001", by_track[1]
        assert by_track[2]["flag"] == "unknown" and by_track[2]["name"] is None
        assert by_track[3]["flag"] == "face_hidden"
        assert any(ev["name"] == "Ali" for ev in s["events"])
    finally:
        e.close()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print("PASS ", t.__name__)
            passed += 1
        except Exception:
            import traceback
            print("FAIL ", t.__name__)
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
