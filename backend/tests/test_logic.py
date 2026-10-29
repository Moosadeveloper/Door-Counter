"""
Unit tests for the door-crossing logic and identity resolution.
Run:  python -m tests.test_logic      (or:  pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AnalysisConfig                        # noqa: E402
from events import Obs, analyze_person                   # noqa: E402
from geometry import DoorLine                            # noqa: E402
from identity import Tracklet, resolve_identities        # noqa: E402

FPS = 10
W, H = 640, 360
DOOR = DoorLine((0.5, 0.0, 0.5, 1.0), W, H)              # vertical line, inside = LEFT (x < 320)


def obs_from_x(xs, start=0, tid=1, y=200):
    out = []
    for i, x in enumerate(xs):
        d, t = DOOR.locate(x, y)
        out.append(Obs(start + i, d, t, tid))
    return out


def analyze(obs):
    return analyze_person(1, obs, DOOR.zone, confirm_n=3)


# ----------------------------------------------------------------- geometry
def test_inside_is_left_for_default_line():
    assert DOOR.locate(100, 50)[0] > 0
    assert DOOR.locate(500, 50)[0] < 0
    flipped = DoorLine((0.5, 0.0, 0.5, 1.0), W, H, flip=True)
    assert flipped.locate(100, 50)[0] < 0


# ------------------------------------------------------------ door crossing
def test_entry_then_exit_and_dwell():
    xs = [560] * 10 + list(np.linspace(560, 100, 30)) + [100] * 50 + list(np.linspace(100, 560, 30)) + [560] * 10
    tl = analyze(obs_from_x(xs))
    kinds = [e.kind for e in tl.events]
    assert kinds == ["entry", "exit"], kinds
    assert len(tl.visits) == 1 and tl.visits[0].start_kind == "door"
    assert tl.events[1].dwell_frames == tl.events[1].frame - tl.events[0].frame
    assert tl.final_side == -1


def test_person_already_inside_then_exits():
    xs = [100] * 20 + list(np.linspace(100, 560, 30)) + [560] * 10
    tl = analyze(obs_from_x(xs))
    assert [e.kind for e in tl.events] == ["exit"]
    assert tl.visits[0].start_kind == "seen_inside"


def test_still_inside_at_end():
    xs = [560] * 10 + list(np.linspace(560, 100, 30)) + [100] * 20
    tl = analyze(obs_from_x(xs))
    assert [e.kind for e in tl.events] == ["entry"]
    assert tl.visits[-1].end_frame is None


def test_reentry_gives_two_visits():
    xs = ([560] * 5 + list(np.linspace(560, 100, 20)) + [100] * 10 + list(np.linspace(100, 560, 20))
          + [560] * 10 + list(np.linspace(560, 100, 20)) + [100] * 10)
    tl = analyze(obs_from_x(xs))
    assert [e.kind for e in tl.events] == ["entry", "exit", "entry"]
    assert len(tl.visits) == 2


def test_jitter_around_line_makes_no_event():
    xs = [330, 312, 328, 315, 331, 310] * 10          # wobbles +-10 px around x=320
    tl = analyze(obs_from_x(xs))
    assert tl is None or not tl.events


def test_single_frame_flicker_is_ignored():
    xs = [560] * 10 + [100] + [560] * 10               # one bad detection on the other side
    tl = analyze(obs_from_x(xs))
    assert not tl.events


def test_walking_past_far_from_door_segment_is_ignored():
    # short door segment on the top half only (y 0..90); person crosses the
    # line's EXTENSION down at y = 300 -> not a door crossing
    door = DoorLine((0.5, 0.0, 0.5, 0.25), W, H)
    obs = []
    for i, x in enumerate(list(np.linspace(560, 100, 40))):
        d, t = door.locate(x, 300)
        obs.append(Obs(i, d, t, 1))
    tl = analyze_person(1, obs, door.zone, confirm_n=3)
    assert tl is not None and not tl.events
    obs = []
    for i, x in enumerate(list(np.linspace(560, 100, 40))):
        d, t = door.locate(x, 40)                      # same walk but through the door segment
        obs.append(Obs(i, d, t, 1))
    tl = analyze_person(1, obs, door.zone, confirm_n=3)
    assert [e.kind for e in tl.events] == ["entry"]


# --------------------------------------------------------------- identities
def make_tracklet(tid, frames, xs, y=300, h=100, face=None, noise_seed=0):
    t = Tracklet(tid)
    for f, x in zip(frames, xs):
        t.add(f, (x - 20, y - h, x + 20, y))
    if face is not None:
        rng = np.random.RandomState(noise_seed)
        for _ in range(3):
            v = face + 0.05 * rng.randn(128).astype(np.float32)
            t.add_face(0.9, v / np.linalg.norm(v), None, 8)
    return t


def unit(seed):
    v = np.random.RandomState(seed).randn(128).astype(np.float32)
    return v / np.linalg.norm(v)


CFG = AnalysisConfig()


def test_same_face_merges_across_id_switch_and_long_gap():
    A, B = unit(1), unit(2)
    t1 = make_tracklet(1, range(0, 40), np.linspace(560, 300, 40), face=A)
    t2 = make_tracklet(7, range(200, 240), np.linspace(150, 100, 40), face=A, noise_seed=1)   # 16 s later
    tb = make_tracklet(3, range(10, 60), [500] * 50, face=B)
    res = resolve_identities({1: t1, 7: t2, 3: tb}, DOOR, FPS, CFG)
    assert res.person_of_track[1] == res.person_of_track[7]
    assert res.person_of_track[3] != res.person_of_track[1]
    assert res.stats["merged_by_face"] == 1 and res.stats["people"] == 2


def test_different_faces_never_merge_even_if_motion_fits_perfectly():
    A, B = unit(1), unit(2)
    t1 = make_tracklet(1, range(0, 40), np.linspace(560, 400, 40), face=A)
    t2 = make_tracklet(2, range(42, 80), np.linspace(400, 300, 38), face=B)      # perfect continuation
    res = resolve_identities({1: t1, 2: t2}, DOOR, FPS, CFG)
    assert res.person_of_track[1] != res.person_of_track[2]


def test_no_face_short_gap_is_linked_by_motion():
    t1 = make_tracklet(1, range(0, 40), np.linspace(560, 400, 40))
    t2 = make_tracklet(2, range(45, 90), np.linspace(395, 300, 45))              # 0.5 s later, same place
    res = resolve_identities({1: t1, 2: t2}, DOOR, FPS, CFG)
    assert res.person_of_track[1] == res.person_of_track[2]
    assert res.stats["merged_by_motion"] == 1


def test_motion_link_rejects_far_away_or_too_late():
    t1 = make_tracklet(1, range(0, 40), np.linspace(560, 400, 40))
    far = make_tracklet(2, range(45, 90), np.linspace(60, 30, 45))
    late = make_tracklet(3, range(120, 160), np.linspace(395, 300, 40))          # 8 s later
    res = resolve_identities({1: t1, 2: far, 3: late}, DOOR, FPS, CFG)
    assert len({res.person_of_track[i] for i in (1, 2, 3)}) == 3


def test_overlapping_tracklets_are_never_merged_even_with_same_face():
    A = unit(1)
    t1 = make_tracklet(1, range(0, 60), [400] * 60, face=A)
    t2 = make_tracklet(2, range(10, 70), [420] * 60, face=A, noise_seed=3)       # same time!
    res = resolve_identities({1: t1, 2: t2}, DOOR, FPS, CFG)
    assert res.person_of_track[1] != res.person_of_track[2]


def test_person_leaving_and_stranger_arriving_are_not_glued_across_the_door():
    # A walks OUT (inside -> outside) and vanishes right after the line.
    a = make_tracklet(1, range(0, 40), np.linspace(200, 345, 40))
    # 0.5 s later a DIFFERENT person appears just inside the line, walking out too.
    b = make_tracklet(2, range(45, 85), np.linspace(300, 450, 40))
    res = resolve_identities({1: a, 2: b}, DOOR, FPS, CFG)
    # b starts at x=300 (inside side, d>0) after a ended at x=345 (outside):
    # implied crossing is "outside -> inside" but A was heading outwards -> blocked
    assert res.person_of_track[1] != res.person_of_track[2]


def test_person_hidden_by_door_frame_is_linked_across_the_line():
    # walks in: last seen just outside, re-appears just inside 0.4 s later, same direction
    a = make_tracklet(1, range(0, 40), np.linspace(560, 340, 40))
    b = make_tracklet(2, range(44, 90), np.linspace(300, 120, 46))
    res = resolve_identities({1: a, 2: b}, DOOR, FPS, CFG)
    assert res.person_of_track[1] == res.person_of_track[2]


def test_short_noise_tracks_are_dropped():
    good = make_tracklet(1, range(0, 40), np.linspace(560, 300, 40))
    noise = make_tracklet(2, range(5, 8), [100] * 3)
    res = resolve_identities({1: good, 2: noise}, DOOR, FPS, CFG)
    assert 2 not in res.person_of_track and res.stats["dropped_short"] == 1


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
