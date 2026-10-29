"""
End-to-end test of process_video() on a synthetic restaurant scene:
real video reading, identity resolution, door events, annotated video
rendering + H.264 conversion, JSON/CSV output. Only YOLO and the face
models are replaced by fakes.

Run:  python -m tests.test_pipeline
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AnalysisConfig                       # noqa: E402
from render import find_ffmpeg                          # noqa: E402
from tests.fakes import Actor, FakeDetector, FakeFaceEngine, Scene, write_scene_video   # noqa: E402
from video_processor import process_video               # noqa: E402

# vertical door line at x = 320, INSIDE = left (x < 320). Entering = walking right -> left.
A = Actor("A", (40, 40, 220),                                   # enters ~5.5s, leaves ~22.8s, tracker loses him twice
          [(0, 560, 300), (20, 560, 300), (80, 150, 300), (200, 150, 300), (260, 560, 300), (300, 560, 300)],
          [(0, 70, 1), (75, 150, 5), (161, 300, 9)])
B = Actor("B", (220, 60, 60),                                   # enters ~7.7s and stays until the end
          [(30, 600, 280), (110, 120, 280), (319, 120, 280)], [(30, 319, 2)])
C = Actor("C", (60, 200, 60),                                   # walks along the outside wall, never enters
          [(0, 600, 200), (319, 600, 340)], [(0, 319, 3)], face_frames=[])
D = Actor("D", (40, 200, 220),                                  # already inside, leaves ~13s, face never visible
          [(0, 100, 250), (100, 100, 250), (160, 500, 250), (319, 500, 250)], [(0, 319, 4)], face_frames=[])
SCENE = Scene([A, B, C, D])


def run_pipeline(use_faces=True):
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "in.mp4")
    write_scene_video(src, SCENE)
    out = os.path.join(tmp, "out")
    cfg = AnalysisConfig(use_faces=use_faces)
    progress = []
    summary, video = process_video(src, out, config=cfg, detector=FakeDetector(SCENE),
                                   face_engine=FakeFaceEngine() if use_faces else None,
                                   progress_cb=lambda p, s: progress.append(p))
    return tmp, out, summary, video, progress


def person_with_track(summary, tid):
    return next(r for r in summary["records"] if tid in r["track_ids"])


def test_full_pipeline_with_faces():
    tmp, out, s, video, progress = run_pipeline(True)
    try:
        assert s["total_entered"] == 2 and s["total_exited"] == 2, (s["total_entered"], s["total_exited"])
        assert s["currently_inside"] == 1
        assert s["unique_people"] == 4 and s["passersby"] == 1 and s["already_inside"] == 1
        assert s["identity"]["raw_tracks"] == 6 and s["identity"]["people"] == 4
        assert s["identity"]["merged_by_face"] == 2

        a = person_with_track(s, 1)                     # tracker IDs 1, 5, 9 -> ONE person
        assert sorted(a["track_ids"]) == [1, 5, 9] and a["status"] == "exited" and a["identity"] == "face"
        assert a["entries"] == 1 and a["exits"] == 1
        assert 15.5 <= a["dwell_seconds"] <= 19.0, a["dwell_seconds"]      # true: ~17.3 s

        b = person_with_track(s, 2)
        assert b["status"] == "inside" and b["entries"] == 1

        d = person_with_track(s, 4)
        assert d["entry_time"] == "Already inside" and d["exit_time"] != "Still Inside" and d["identity"] == "tracking"

        # exit of A must be attributed to A even though a different tracker ID saw it
        ev = [e for e in s["events"] if e["person_id"] == a["person_id"]]
        assert [e["type"] for e in ev] == ["entry", "exit"]
        assert ev[1]["face_verified"] and ev[1]["dwell_time"] not in (None, "")
        # entry frame ~55, exit ~228 (from the scripted walk)
        assert abs(ev[0]["frame"] - 55) <= 4 and abs(ev[1]["frame"] - 228) <= 4, [e["frame"] for e in ev]

        # occupancy timeline ends where currently_inside says
        # occupancy: D already inside (1) -> A enters (2) -> B enters (3) -> D leaves (2) -> A leaves (1)
        assert [p["inside"] for p in s["timeline"]] == [1, 2, 3, 2, 1, 1] or \
               [p["inside"] for p in s["timeline"]][:5] == [1, 2, 3, 2, 1], s["timeline"]
        assert s["timeline"][-1]["inside"] == s["currently_inside"] == 1

        # files
        for f in ("results.json", "results.csv", "events.csv", "processed_video.mp4"):
            assert os.path.getsize(os.path.join(out, f)) > 0, f
        assert os.path.exists(os.path.join(out, "faces", f"person_{a['person_id']}.jpg"))
        assert os.path.exists(os.path.join(out, "people", f"person_{d['person_id']}.jpg"))
        json.load(open(os.path.join(out, "results.json")))             # valid JSON

        # progress goes up and ends at 1
        assert progress[-1] == 1.0 and progress == sorted(progress)

        # annotated video: same frame count, and H.264 (browser playable) when ffmpeg exists
        cap = cv2.VideoCapture(video)
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        assert n == SCENE.frames, n
        if find_ffmpeg():
            assert s["video"]["browser_playable"]
            probe = subprocess.run([find_ffmpeg(), "-i", video], capture_output=True, text=True).stderr
            assert "h264" in probe and "yuv420p" in probe, probe[-400:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pipeline_without_faces_still_counts_doors_correctly():
    tmp, out, s, video, _ = run_pipeline(False)
    try:
        # no faces: A's ID switches (0.4 s gap, 1 s gap) are repaired by the motion linker
        assert s["total_entered"] == 2 and s["total_exited"] == 2 and s["currently_inside"] == 1, s
        assert s["identity"]["merged_by_face"] == 0
        assert s["identity"]["people"] <= 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_flip_swaps_entry_and_exit():
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "in.mp4")
        write_scene_video(src, SCENE)
        s, _ = process_video(src, os.path.join(tmp, "o"), config=AnalysisConfig(flip=True, use_faces=False),
                             detector=FakeDetector(SCENE))
        # with the inside on the RIGHT, A and B "exit" the restaurant, D "enters" it
        assert s["total_exited"] == 2 and s["total_entered"] == 2, (s["total_entered"], s["total_exited"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
