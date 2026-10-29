"""
events.py
---------
Turns "where was each PERSON, frame by frame" into entry / exit events,
visits, dwell times and the final results dictionary.

Important: this works on PEOPLE (after identity resolution), not raw tracker
IDs. So if the tracker loses someone and gives them a new ID, but the face
matches, the exit is still logged against the same person.
"""
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional


def format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    return str(timedelta(seconds=int(round(max(0.0, seconds)))))


@dataclass
class Obs:
    """One sighting of a person: frame, signed distance to the door line
    (+ inside), position along the line, and which tracker ID saw it."""
    frame: int
    d: float
    t: float
    tid: int


@dataclass
class Event:
    kind: str            # "entry" | "exit"
    frame: int           # best estimate of the frame the door line was crossed
    tid: int             # tracker ID that saw the crossing
    gap_frames: int      # frames the person was unseen around the crossing
    dwell_frames: Optional[int] = None   # exits only: length of the visit just ended


@dataclass
class Visit:
    start_frame: int
    end_frame: Optional[int]     # None = still inside at the end of the video
    start_kind: str              # "door" | "seen_inside"


@dataclass
class PersonTimeline:
    person_id: int
    first_frame: int
    last_frame: int
    initial_side: Optional[int]
    final_side: Optional[int]
    events: List[Event] = field(default_factory=list)
    visits: List[Visit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Crossing detection for ONE person
# ---------------------------------------------------------------------------
def analyze_person(person_id: int, obs: List[Obs], zone: float, confirm_n: int,
                   span=(-0.25, 1.25), interp_gap_frames: int = 25,
                   initial_n: int = 2) -> Optional[PersonTimeline]:
    """
    Robust line-crossing:
      * dead zone around the line -> tracker jitter cannot fake a crossing
      * a new side must be seen `confirm_n` times in a row before it counts
      * the crossing point must fall on the drawn door segment (a person
        walking past far from the door but crossing its extension is ignored)
    """
    obs = sorted(obs, key=lambda o: o.frame)
    if not obs:
        return None

    events: List[Event] = []
    cur: Optional[int] = None
    initial_side: Optional[int] = None
    cand: Optional[int] = None
    cand_n = 0
    cand_first: Optional[Obs] = None
    last_on_cur: Optional[Obs] = None

    for o in obs:
        if o.d > zone:
            s = 1
        elif o.d < -zone:
            s = -1
        else:
            continue                     # inside the dead zone: ignore

        if s == cur:                     # still on the confirmed side
            cand, cand_n, cand_first = None, 0, None
            last_on_cur = o
            continue

        if s != cand:                    # start (or restart) a candidate streak
            cand, cand_n, cand_first = s, 1, o
        else:
            cand_n += 1

        need = initial_n if cur is None else confirm_n
        if cand_n < need:
            continue

        # ---- the new side is confirmed ---------------------------------
        if cur is None:
            cur = s
            initial_side = s
        else:
            a = last_on_cur.d / (last_on_cur.d - cand_first.d)   # 0..1 crossing point
            t_cross = last_on_cur.t + a * (cand_first.t - last_on_cur.t)
            if span[0] <= t_cross <= span[1]:
                gap = cand_first.frame - last_on_cur.frame
                if gap <= interp_gap_frames:
                    frame = int(round(last_on_cur.frame + a * gap))
                else:
                    frame = cand_first.frame           # person was unseen; use first sighting
                events.append(Event("entry" if s == 1 else "exit", frame,
                                    cand_first.tid, gap))
            cur = s
        last_on_cur = o
        cand, cand_n, cand_first = None, 0, None

    if cur is None:
        return None                       # never left the dead zone: unusable

    tl = PersonTimeline(person_id, obs[0].frame, obs[-1].frame, initial_side, cur, events)
    _build_visits(tl)
    return tl


def _build_visits(tl: PersonTimeline) -> None:
    open_v: Optional[Visit] = None
    if tl.initial_side == 1:
        open_v = Visit(tl.first_frame, None, "seen_inside")
    for ev in tl.events:
        if ev.kind == "entry":
            if open_v is None:
                open_v = Visit(ev.frame, None, "door")
        else:
            if open_v is None:                       # exit without a seen entry
                open_v = Visit(tl.first_frame, None, "seen_inside")
            open_v.end_frame = ev.frame
            ev.dwell_frames = max(0, ev.frame - open_v.start_frame)
            tl.visits.append(open_v)
            open_v = None
    if open_v is not None:
        tl.visits.append(open_v)


# ---------------------------------------------------------------------------
# Final results (JSON / CSV friendly)
# ---------------------------------------------------------------------------
def build_results(timelines: List[PersonTimeline], persons: Dict[int, object],
                  fps: float, end_frame: int, face_match_thr: float,
                  capacity: Optional[int] = None) -> dict:
    """`persons` maps person_id -> identity.PersonIdentity (face info)."""
    end_s = end_frame / fps
    records, events_out, changes = [], [], []
    total_entries = total_exits = already_inside = passersby = 0
    unique_entered = set()

    for tl in sorted(timelines, key=lambda t: t.person_id):
        pid = tl.person_id
        ident = persons.get(pid)
        n_face = getattr(ident, "face_samples", 0) if ident else 0

        for ev in tl.events:
            total_entries += ev.kind == "entry"
            total_exits += ev.kind == "exit"
            sim = ident.tracklet_sim.get(ev.tid) if ident else None
            events_out.append({
                "time": format_time(ev.frame / fps),
                "seconds": round(ev.frame / fps, 2),
                "frame": ev.frame,
                "person_id": pid,
                "type": ev.kind,
                "face_verified": bool(sim is not None and sim >= face_match_thr),
                "similarity": None if sim is None else round(float(sim), 3),
                "inferred": ev.gap_frames > 2 * fps,
                "dwell_time": format_time(ev.dwell_frames / fps) if ev.dwell_frames is not None else None,
                "name": getattr(ident, "name", None),
                "roll_no": getattr(ident, "roll_no", None),
                "flag": getattr(ident, "flag", None),
            })

        if not tl.visits:
            passersby += 1                 # seen only outside, never went in
            continue

        visits_out, dwell = [], 0.0
        for v in tl.visits:
            s = v.start_frame / fps
            e = (v.end_frame / fps) if v.end_frame is not None else end_s
            dwell += max(0.0, e - s)
            changes.append((v.start_frame, +1))
            if v.end_frame is not None:
                changes.append((v.end_frame, -1))
            visits_out.append({"start": round(s, 2),
                               "end": None if v.end_frame is None else round(e, 2),
                               "start_kind": v.start_kind,
                               "dwell_seconds": round(max(0.0, e - s), 2)})
            if v.start_kind == "door":
                unique_entered.add(pid)
        if tl.visits[0].start_kind == "seen_inside":
            already_inside += 1

        first, last = tl.visits[0], tl.visits[-1]
        still_in = last.end_frame is None
        records.append({
            "person_id": pid,
            "entry_time": format_time(first.start_frame / fps) if first.start_kind == "door" else "Already inside",
            "exit_time": "Still Inside" if still_in else format_time(last.end_frame / fps),
            "dwell_time": format_time(dwell),
            "dwell_seconds": round(dwell, 2),
            "status": "inside" if still_in else "exited",
            "entries": sum(1 for e in tl.events if e.kind == "entry"),
            "exits": sum(1 for e in tl.events if e.kind == "exit"),
            "visits": visits_out,
            "first_seen": format_time(tl.first_frame / fps),
            "face_samples": n_face,
            "has_face": bool(ident and ident.best_face_jpg),
            "has_snapshot": bool(ident and ident.snapshot_jpg),
            "identity": "face" if n_face else "tracking",
            "track_ids": list(ident.track_ids) if ident else [],
            "registered_id": getattr(ident, "registered_id", None),
            "name": getattr(ident, "name", None),
            "roll_no": getattr(ident, "roll_no", None),
            "match_score": getattr(ident, "match_score", None),
            "flag": getattr(ident, "flag", None),
        })

    events_out.sort(key=lambda e: e["frame"])

    # people-inside-over-time (step function)
    changes.sort()
    timeline, inside = [{"t": 0.0, "inside": 0}], 0
    for frame, delta in changes:
        inside += delta
        t = round(frame / fps, 2)
        if timeline[-1]["t"] == t:
            timeline[-1]["inside"] = inside
        else:
            timeline.append({"t": t, "inside": inside})
    if timeline[-1]["t"] < end_s:
        timeline.append({"t": round(end_s, 2), "inside": inside})

    analytics = _analytics(records, events_out, timeline, end_s, capacity)

    return {
        "total_entered": total_entries,          # entry EVENTS (door counter)
        "total_exited": total_exits,             # exit EVENTS
        "currently_inside": inside,              # people inside at the end of the video
        "unique_people": len(timelines),
        "unique_entered": len(unique_entered),   # different people who came in
        "already_inside": already_inside,
        "passersby": passersby,
        "records": records,
        "events": events_out,
        "timeline": timeline,
        "analytics": analytics,
    }


def _analytics(records: list, events_out: list, timeline: List[dict], end_s: float,
               capacity: Optional[int]) -> dict:
    """Extra numbers for a results dashboard: busiest moment, average stay,
    busiest arrival window, and (if a capacity was given) how long the room
    was over its limit. All derived from data already computed above -
    nothing here changes the counts themselves."""
    peak_inside, peak_at = 0, 0.0
    for i, pt in enumerate(timeline):
        if pt["inside"] > peak_inside:
            peak_inside, peak_at = pt["inside"], pt["t"]

    dwells = [r["dwell_seconds"] for r in records if r["dwell_seconds"] > 0]
    avg_dwell = round(sum(dwells) / len(dwells), 2) if dwells else 0.0
    max_dwell = round(max(dwells), 2) if dwells else 0.0

    busiest_window = None
    entries = sorted(e["seconds"] for e in events_out if e["type"] == "entry")
    if entries and end_s > 0:
        win = max(30.0, end_s / 8.0)                    # ~8 buckets across the video, 30s minimum
        n_win = max(1, int(end_s // win) + 1)
        counts = [0] * n_win
        for t in entries:
            counts[min(n_win - 1, int(t // win))] += 1
        i = max(range(n_win), key=lambda k: counts[k])
        if counts[i] > 0:
            busiest_window = {"start": round(i * win, 2), "end": round(min(end_s, (i + 1) * win), 2),
                              "entries": counts[i]}

    over_capacity_seconds = 0.0
    over_capacity_periods: List[dict] = []
    if capacity is not None and capacity >= 0:
        for i, pt in enumerate(timeline):
            if pt["inside"] <= capacity:
                continue
            t0 = pt["t"]
            t1 = timeline[i + 1]["t"] if i + 1 < len(timeline) else end_s
            if t1 > t0:
                over_capacity_seconds += t1 - t0
                if over_capacity_periods and over_capacity_periods[-1]["end"] >= t0 - 1e-6:
                    over_capacity_periods[-1]["end"] = t1
                else:
                    over_capacity_periods.append({"start": round(t0, 2), "end": round(t1, 2)})

    return {
        "peak_inside": peak_inside,
        "peak_at": round(peak_at, 2),
        "avg_dwell_seconds": avg_dwell,
        "max_dwell_seconds": max_dwell,
        "busiest_window": busiest_window,
        "capacity": capacity,
        "over_capacity_seconds": round(over_capacity_seconds, 2),
        "over_capacity_periods": over_capacity_periods,
    }
