"""
config.py
---------
Every tunable number lives here, so you can experiment (and explain your
choices in the viva) without hunting through the code.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class AnalysisConfig:
    # ---- door line (normalised 0..1 coordinates: x1, y1, x2, y2) ----------
    # Default = a vertical line in the middle of the frame, "inside" on the
    # left side (same behaviour as the old version: right -> left = entry).
    line: Tuple[float, float, float, float] = (0.5, 0.0, 0.5, 1.0)
    flip: bool = False          # True = swap which side counts as "inside"
    anchor: str = "foot"        # "foot" (bottom-centre of box) or "center"

    # ---- detection / tracking ---------------------------------------------
    stride: int = 1             # 1 = every frame, 2 = every 2nd frame, ...
    conf: float = 0.25          # YOLO confidence threshold
    iou: float = 0.7            # NMS overlap threshold. Higher = two people standing/walking
                                 # close together are more likely to stay as two separate boxes
                                 # instead of being merged into one by non-max-suppression.
    imgsz: int = 768             # bumped from 640: more pixels per person -> boxes for two
                                 # close-together people separate more cleanly. Use 960 in the
                                 # UI's "Detail" setting for very wide/crowded CCTV shots.
    use_faces: bool = True      # face re-identification on/off
    capacity: Optional[int] = None   # max people allowed inside; None = no alert

    # ---- door crossing logic ----------------------------------------------
    zone_frac: float = 0.012          # dead-zone half-width (fraction of frame size)
    confirm_seconds: float = 0.35     # a side must be seen this long to be accepted
    span_margin: float = 0.25         # crossing may miss the drawn segment by 25 %
    interp_gap_seconds: float = 1.0   # interpolate crossing time if gap <= this
    inferred_gap_seconds: float = 2.0 # crossing while unseen longer than this = "inferred"

    # ---- noise filter -----------------------------------------------------
    min_obs: int = 5                  # ignore tracks seen in fewer frames
    min_track_seconds: float = 0.4    # ... or shorter than this

    # ---- faces ------------------------------------------------------------
    face_every_seconds: float = 0.4   # try a face crop this often per person
    max_faces_per_track: int = 8
    min_person_height_px: int = 80    # smaller people: face is too tiny to use
    min_face_px: int = 24
    face_det_score: float = 0.7
    face_match_thr: float = 0.363     # SFace's own recommended cosine threshold (opencv_zoo).
                                       # Was 0.40 - too strict, so a person who left and came back
                                       # (new face samples, slightly different angle/light) often
                                       # failed to match their own earlier face and got a new ID.
    face_conflict_thr: float = 0.15   # below this -> definitely different people

    # ---- motion based ID linking (fallback when there is no face) ---------
    link_gap_seconds: float = 3.0     # max time a person may vanish and be re-linked
    color_block: float = 0.75         # clothing-colour distance above this = different
    overlap_tolerance_seconds: float = 0.4

    def normalise(self) -> "AnalysisConfig":
        self.stride = max(1, int(self.stride))
        self.anchor = "center" if self.anchor == "center" else "foot"
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in self.line)
        self.line = (x1, y1, x2, y2)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["line"] = list(self.line)
        return d


@dataclass
class InstitutionConfig:
    """Settings for the institution / attendance part (registered people,
    live camera, unknown-person alerts). Change them here."""
    # ---- matching a face against REGISTERED people ------------------------
    recog_thr: float = 0.40          # cosine similarity needed to say "this is <name>".
                                     # Stricter than face_match_thr (0.363) on purpose: giving
                                     # attendance to the WRONG student is worse than asking
                                     # for one more look at the face.
    recog_margin: float = 0.04       # best match must beat the 2nd-best PERSON by this much
    votes_needed: int = 2            # this many face sightings must agree before a name is given
    duplicate_thr: float = 0.55      # while registering: a new photo this similar to someone
                                     # ELSE already registered = probably the same human

    # ---- live camera ------------------------------------------------------
    face_every_seconds: float = 0.25 # how often to look at each person's face
    unknown_after_seconds: float = 3.0   # face seen, no match for this long -> "UNKNOWN" alert
    hidden_after_seconds: float = 4.0    # person seen, face NEVER visible this long -> "FACE HIDDEN" alert
    min_faces_for_unknown: int = 3       # need at least this many face looks before calling someone unknown
    min_alert_height_px: int = 120       # people smaller than this are too far away to judge
    forget_track_seconds: float = 5.0    # drop a person from memory after not seeing them this long
    log_cooldown_seconds: float = 60.0   # same person, same camera: log again only after this gap
    live_imgsz: int = 640
    live_conf: float = 0.35
    confirm_frames: int = 3              # door-line mode: frames on the new side before it counts

    # ---- STORAGE SAVER (snapshot photos) ----------------------------------
    snap_max_px: int = 320               # snapshot ki lambi side isse bari ho to chhoti kar do
    snap_quality: int = 60               # JPEG quality (60 = choti file, phir bhi saaf dikhti hai)
    snap_known: str = "first"            # registered log: "all" = har in/out ki photo,
                                         #   "first" = din me sirf pehli photo, "none" = koi photo nahi
    snap_keep_days: int = 15             # is se purane din ke snapshot folders khud delete (0 = kabhi nahi)
    registry_max_px: int = 800           # register ki gayi photos bhi is size tak chhoti
    registry_quality: int = 80
