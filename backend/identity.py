"""
identity.py
-----------
Who is who?

ByteTrack gives every "track" an ID, but the same human often ends up with
several IDs (walks behind a pillar, door frame hides them, tracker gets
confused). Here we glue those fragments ("tracklets") back into PEOPLE:

  1. FACE   - tracklets whose face fingerprints match are the same person.
  2. MOTION - a track that ends and another that starts right there a moment
              later (and heads the same way) are linked. Fallback for people
              whose face was never visible.

Hard rule for both: one person cannot be in two places at the same time, so
two tracklets that overlap in time are never merged.

The video is processed offline, so we can resolve identities with the WHOLE
video in view (much more accurate than deciding frame by frame).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from events import Obs


# ---------------------------------------------------------------------------
# Tracklet: everything we learn about one tracker ID
# ---------------------------------------------------------------------------
class Tracklet:
    def __init__(self, track_id: int):
        self.track_id = int(track_id)
        self.frames: List[int] = []
        self.boxes: List[Tuple[float, float, float, float]] = []
        self.faces: List[Tuple[float, np.ndarray]] = []       # (quality, vec), best first
        self.best_face_img: Optional[Tuple[float, bytes]] = None
        self.snapshot: Optional[Tuple[float, np.ndarray]] = None   # (box area, BGR crop)
        self._color_sum: Optional[np.ndarray] = None
        self._color_n = 0
        self.last_face_try = -10 ** 9
        self.last_color_try = -10 ** 9

    # -- basic info ---------------------------------------------------------
    def add(self, frame_idx: int, box) -> None:
        self.frames.append(int(frame_idx))
        self.boxes.append(tuple(float(v) for v in box))

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def n_obs(self) -> int:
        return len(self.frames)

    def anchor(self, i: int, mode: str) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.boxes[i]
        return (x1 + x2) / 2.0, (y2 if mode == "foot" else (y1 + y2) / 2.0)

    def obs_series(self, door, mode: str) -> List[Obs]:
        out = []
        for i, f in enumerate(self.frames):
            x, y = self.anchor(i, mode)
            d, t = door.locate(x, y)
            out.append(Obs(f, d, t, self.track_id))
        return out

    # -- faces --------------------------------------------------------------
    def add_face(self, quality: float, vec: np.ndarray, thumb_jpg: Optional[bytes], keep: int) -> None:
        self.faces.append((float(quality), vec))
        self.faces.sort(key=lambda q: -q[0])
        del self.faces[keep:]
        if thumb_jpg is not None and (self.best_face_img is None or quality > self.best_face_img[0]):
            self.best_face_img = (float(quality), thumb_jpg)

    def face_sum(self) -> Optional[np.ndarray]:
        if not self.faces:
            return None
        return np.sum([q * v for q, v in self.faces], axis=0).astype(np.float32)

    def face_centroid(self) -> Optional[np.ndarray]:
        s = self.face_sum()
        if s is None:
            return None
        n = float(np.linalg.norm(s))
        return s / n if n > 1e-9 else None

    # -- clothing colour ----------------------------------------------------
    def add_color(self, hist: np.ndarray) -> None:
        self._color_sum = hist if self._color_sum is None else self._color_sum + hist
        self._color_n += 1

    def color(self) -> Optional[np.ndarray]:
        if self._color_sum is None:
            return None
        s = self._color_sum / max(1e-9, float(self._color_sum.sum()))
        return s.astype(np.float32)


def torso_hist(frame: np.ndarray, box) -> Optional[np.ndarray]:
    """Colour fingerprint of the torso (hue x saturation histogram)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1, tx2 = int(max(0, x1 + 0.25 * bw)), int(min(W, x2 - 0.25 * bw))
    ty1, ty2 = int(max(0, y1 + 0.30 * bh)), int(min(H, y1 + 0.65 * bh))
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    total = float(h.sum())
    return (h / total).astype(np.float32) if total > 0 else None


def _color_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Bhattacharyya distance: 0 = identical colours, 1 = nothing in common."""
    if a is None or b is None:
        return None
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class PersonIdentity:
    person_id: int
    track_ids: List[int]
    start: int
    end: int
    centroid: Optional[np.ndarray] = None
    face_samples: int = 0
    best_face_jpg: Optional[bytes] = None
    snapshot_jpg: Optional[bytes] = None
    tracklet_sim: Dict[int, float] = field(default_factory=dict)   # track_id -> similarity to person
    # filled in only when people are registered (institution mode)
    registered_id: Optional[str] = None
    name: Optional[str] = None
    roll_no: Optional[str] = None
    match_score: Optional[float] = None
    flag: Optional[str] = None          # None | "unknown" (face seen, not registered) | "face_hidden"


@dataclass
class IdentityResult:
    persons: Dict[int, PersonIdentity]
    person_of_track: Dict[int, int]
    stats: dict


# ---------------------------------------------------------------------------
# Cluster = a group of tracklets believed to be one person
# ---------------------------------------------------------------------------
class _Cluster:
    def __init__(self, tl: Tracklet):
        self.tracklets = [tl]
        self.vec_sum = tl.face_sum()
        self.start, self.end = tl.start, tl.end
        self.first_tl, self.last_tl = tl, tl

    def absorb(self, other: "_Cluster") -> None:
        self.tracklets += other.tracklets
        if other.vec_sum is not None:
            self.vec_sum = other.vec_sum if self.vec_sum is None else self.vec_sum + other.vec_sum
        if other.start < self.start:
            self.start, self.first_tl = other.start, other.first_tl
        if other.end > self.end:
            self.end, self.last_tl = other.end, other.last_tl

    def centroid(self) -> Optional[np.ndarray]:
        if self.vec_sum is None:
            return None
        n = float(np.linalg.norm(self.vec_sum))
        return (self.vec_sum / n).astype(np.float32) if n > 1e-9 else None

    def color(self) -> Optional[np.ndarray]:
        cols = [(t.color(), t.n_obs()) for t in self.tracklets if t.color() is not None]
        if not cols:
            return None
        s = sum(c * w for c, w in cols)
        return (s / max(1e-9, float(s.sum()))).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1: merge by face
# ---------------------------------------------------------------------------
def _merge_by_face(clusters, alive, conflict, cfg) -> int:
    n = len(clusters)
    D = next((c.vec_sum.shape[0] for c in clusters if c.vec_sum is not None), None)
    if D is None:
        return 0
    C = np.zeros((n, D), np.float32)
    has = np.zeros(n, bool)
    for i, c in enumerate(clusters):
        cen = c.centroid()
        if cen is not None:
            C[i], has[i] = cen, True

    valid = has[:, None] & has[None, :] & ~conflict
    np.fill_diagonal(valid, False)
    S = np.where(valid, C @ C.T, -1.0)

    merges = 0
    while True:
        k = int(np.argmax(S))
        i, j = divmod(k, n)
        if S[i, j] < cfg.face_match_thr:
            break
        if i > j:
            i, j = j, i
        clusters[i].absorb(clusters[j])                 # merge j into i
        alive[j] = False
        conflict[i, :] |= conflict[j, :]
        conflict[:, i] = conflict[i, :]
        conflict[i, i] = False
        has[j] = False
        C[i] = clusters[i].centroid()
        row = C @ C[i]
        ok = has & ~conflict[i, :]
        ok[i] = False
        row = np.where(ok, row, -1.0)
        S[i, :], S[:, i] = row, row
        S[j, :], S[:, j] = -1.0, -1.0
        merges += 1
    return merges


# ---------------------------------------------------------------------------
# Step 2: merge by motion (fallback for face-less people)
# ---------------------------------------------------------------------------
@dataclass
class _Endpoint:
    x: float
    y: float
    h: float
    d: float
    t: float
    v: float          # speed of signed distance (pixels / frame), + = heading inside


def _tracklet_endpoints(tl: Tracklet, door, mode: str) -> Tuple[_Endpoint, _Endpoint]:
    obs = tl.obs_series(door, mode)

    def make(i: int, window: List[Obs]) -> _Endpoint:
        x, y = tl.anchor(i, mode)
        x1, y1, x2, y2 = tl.boxes[i]
        df = window[-1].frame - window[0].frame
        v = (window[-1].d - window[0].d) / df if df > 0 else 0.0
        return _Endpoint(x, y, max(1.0, y2 - y1), obs[i].d, obs[i].t, v)

    k = min(6, len(obs))
    start = make(0, obs[:k])
    end = make(len(obs) - 1, obs[-k:])
    return start, end


def _side(d: float, zone: float) -> int:
    return 1 if d > zone else (-1 if d < -zone else 0)


def _link_cost(ca: _Cluster, cb: _Cluster, ends, door, fps, cfg, gap: int) -> Optional[float]:
    a = ends[ca.last_tl.track_id][1]       # where A disappeared
    b = ends[cb.first_tl.track_id][0]      # where B appeared
    dist = math.hypot(a.x - b.x, a.y - b.y)
    H = max(20.0, 0.5 * (a.h + b.h))
    gap_s = gap / fps
    allow = 0.5 * H + 1.0 * H * gap_s      # a walking person moves ~1 body-height per second
    if dist > allow:
        return None

    fa, fb = ca.centroid(), cb.centroid()
    face_sim = None
    if fa is not None and fb is not None:
        face_sim = float(fa @ fb)
        if face_sim < cfg.face_conflict_thr:          # faces clearly differ
            return None

    col = _color_dist(ca.color(), cb.color())
    if col is not None and col > cfg.color_block:      # clothes clearly differ
        return None

    # Linking ACROSS the door line is the risky case (someone leaves, someone
    # else arrives at the same spot). Only allow it if A was really heading
    # toward the other side and B keeps going the same way.
    sa, sb = _side(a.d, door.zone), _side(b.d, door.zone)
    if sa and sb and sa != sb:
        if a.v * sb <= 0 or b.v * sb < 0:
            return None
        alpha = a.d / (a.d - b.d)
        t_cross = a.t + alpha * (b.t - a.t)
        if not (-cfg.span_margin <= t_cross <= 1 + cfg.span_margin):
            return None

    cost = dist / allow + 0.15 * gap / max(1.0, cfg.link_gap_seconds * fps)
    cost += 0.6 * col if col is not None else 0.2
    if face_sim is not None:
        cost -= 0.4 * max(0.0, face_sim - cfg.face_conflict_thr)
    return cost


def _merge_by_motion(clusters, alive, conflict, ends, door, fps, cfg) -> int:
    max_gap = cfg.link_gap_seconds * fps
    if max_gap <= 0:
        return 0
    total = 0
    for _ in range(10):                                   # chains: A->B->C need several rounds
        idx = [i for i in range(len(clusters)) if alive[i]]
        cands = []
        for a in idx:
            ca = clusters[a]
            for b in idx:
                if a == b:
                    continue
                cb = clusters[b]
                gap = cb.start - ca.end
                if gap <= 0 or gap > max_gap or conflict[a, b]:
                    continue
                cost = _link_cost(ca, cb, ends, door, fps, cfg, gap)
                if cost is not None:
                    cands.append((cost, a, b))
        if not cands:
            break
        cands.sort()
        used = set()
        chosen = []
        for _, a, b in cands:                             # cheapest first, one link per cluster
            if a in used or b in used:
                continue
            used.update((a, b))
            chosen.append((a, b))
        for a, b in chosen:
            clusters[a].absorb(clusters[b])
            alive[b] = False
            conflict[a, :] |= conflict[b, :]
            conflict[:, a] = conflict[a, :]
            conflict[a, a] = False
            total += 1
    return total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def resolve_identities(tracklets: Dict[int, Tracklet], door, fps: float, cfg) -> IdentityResult:
    n_raw = len(tracklets)
    min_dur = cfg.min_track_seconds * fps
    kept = [t for t in tracklets.values()
            if t.n_obs() >= cfg.min_obs and (t.end - t.start) >= min_dur]
    kept.sort(key=lambda t: t.start)
    stats = {"raw_tracks": n_raw, "kept_tracks": len(kept), "dropped_short": n_raw - len(kept),
             "people": 0, "merged_by_face": 0, "merged_by_motion": 0}
    if not kept:
        return IdentityResult({}, {}, stats)

    n = len(kept)
    tol = cfg.overlap_tolerance_seconds * fps
    starts = np.array([t.start for t in kept], float)
    ends_f = np.array([t.end for t in kept], float)
    conflict = (np.minimum(ends_f[:, None], ends_f[None, :])
                - np.maximum(starts[:, None], starts[None, :])) > tol
    np.fill_diagonal(conflict, False)

    clusters = [_Cluster(t) for t in kept]
    alive = [True] * n

    stats["merged_by_face"] = _merge_by_face(clusters, alive, conflict, cfg)
    ends = {t.track_id: _tracklet_endpoints(t, door, cfg.anchor) for t in kept}
    stats["merged_by_motion"] = _merge_by_motion(clusters, alive, conflict, ends, door, fps, cfg)

    final = sorted((clusters[i] for i in range(n) if alive[i]), key=lambda c: c.start)
    persons: Dict[int, PersonIdentity] = {}
    person_of_track: Dict[int, int] = {}

    for pid, c in enumerate(final, start=1):
        c.tracklets.sort(key=lambda t: t.start)
        cen = c.centroid()
        best_face = max((t.best_face_img for t in c.tracklets if t.best_face_img), default=None,
                        key=lambda q: q[0])
        best_snap = max((t.snapshot for t in c.tracklets if t.snapshot), default=None,
                        key=lambda q: q[0])
        snap_jpg = None
        if best_snap is not None:
            ok, buf = cv2.imencode(".jpg", best_snap[1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            snap_jpg = buf.tobytes() if ok else None

        sims = {}
        for t in c.tracklets:
            person_of_track[t.track_id] = pid
            tc = t.face_centroid()
            if tc is not None and cen is not None:
                sims[t.track_id] = float(tc @ cen)

        persons[pid] = PersonIdentity(
            person_id=pid,
            track_ids=[t.track_id for t in c.tracklets],
            start=c.start, end=c.end,
            centroid=cen,
            face_samples=sum(len(t.faces) for t in c.tracklets),
            best_face_jpg=best_face[1] if best_face else None,
            snapshot_jpg=snap_jpg,
            tracklet_sim=sims,
        )

    stats["people"] = len(persons)
    return IdentityResult(persons, person_of_track, stats)
