"""
registry.py
-----------
The list of REGISTERED people (students, teachers, staff) and their face
fingerprints.

"Training" here means ENROLMENT - the standard way real attendance systems
work: every photo you upload (front, left side, right side, ...) is turned
into a 128-number SFace fingerprint and saved. Later, a face seen by the
camera is compared with every saved fingerprint. No neural network has to be
re-trained, so adding a new student takes one second, not hours.

Why 3 photos? A face turned to the side gives a different fingerprint than a
straight face. Storing front + left + right (plus a mirrored copy of each)
means a person walking in at an angle is still close to one of their stored
samples.

Everything is plain files:
    data/registry/people.json               names, roll numbers, photo list
    data/registry/<person_id>/photo_*.jpg   the uploaded photos
    data/registry/<person_id>/face_*.jpg    the aligned face that was used
    data/registry/<person_id>/emb.npy       fingerprints (N x 128)
"""
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ANGLES = ("front", "left", "right", "extra")


class EnrollError(ValueError):
    """Shown to the user as-is (bad photo, duplicate roll number, ...)."""


@dataclass
class Match:
    person_id: str
    name: str
    roll_no: str
    score: float
    second: float            # best score of any OTHER person (-1 if none)


def decode_image(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


class PersonRegistry:
    def __init__(self, root: str, max_px: int = 800, quality: int = 80):
        self.root = root
        self.max_px, self.quality = max_px, quality      # storage saver for saved photos
        os.makedirs(root, exist_ok=True)
        self._lock = threading.RLock()
        self.people: Dict[str, dict] = {}
        self._emb: Dict[str, np.ndarray] = {}
        self._matrix: Optional[np.ndarray] = None     # all fingerprints stacked
        self._owner: List[str] = []                   # row -> person_id
        self.load()

    # ------------------------------------------------------------------ files
    @property
    def _index_path(self) -> str:
        return os.path.join(self.root, "people.json")

    def _folder(self, pid: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{8}", pid or ""):
            raise KeyError(pid)
        return os.path.join(self.root, pid)

    def load(self) -> None:
        with self._lock:
            self.people, self._emb = {}, {}
            if os.path.isfile(self._index_path):
                with open(self._index_path, encoding="utf-8") as fh:
                    for p in json.load(fh):
                        self.people[p["id"]] = p
            for pid in list(self.people):
                path = os.path.join(self._folder(pid), "emb.npy")
                if os.path.isfile(path):
                    self._emb[pid] = np.load(path).astype(np.float32)
            self._rebuild()

    def _save_index(self) -> None:
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(sorted(self.people.values(), key=lambda p: p["name"].lower()), fh, indent=2)
        os.replace(tmp, self._index_path)

    def _rebuild(self) -> None:
        rows, owner = [], []
        for pid, E in self._emb.items():
            if pid in self.people and E is not None and len(E):
                rows.append(E)
                owner += [pid] * len(E)
        self._matrix = np.vstack(rows).astype(np.float32) if rows else None
        self._owner = owner

    # ------------------------------------------------------------------ read
    def count(self) -> int:
        return len(self.people)

    def list(self) -> List[dict]:
        with self._lock:
            return sorted((dict(p) for p in self.people.values()), key=lambda p: p["name"].lower())

    def get(self, pid: str) -> Optional[dict]:
        with self._lock:
            p = self.people.get(pid)
            return dict(p) if p else None

    def photo_path(self, pid: str, filename: str) -> Optional[str]:
        if not re.fullmatch(r"(photo|face)_\d+\.jpg", filename or ""):
            return None
        try:
            path = os.path.join(self._folder(pid), filename)
        except KeyError:
            return None
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------ match
    def match(self, vec: np.ndarray, thr: float, margin: float = 0.0) -> Optional[Match]:
        """Best registered person for one face fingerprint, or None."""
        scores = self.scores(vec)
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        pid, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if best < thr or (second >= 0 and best - second < margin):
            return None
        p = self.people.get(pid)
        if p is None:
            return None
        return Match(pid, p["name"], p.get("roll_no", ""), float(best), float(second))

    def scores(self, vec: np.ndarray) -> Dict[str, float]:
        """person_id -> highest similarity to ANY of that person's stored samples.
        (max, not average: a side-face should be compared with the side photo.)"""
        with self._lock:
            M, owner = self._matrix, self._owner
        if M is None or vec is None:
            return {}
        sims = M @ np.asarray(vec, np.float32).reshape(-1)
        out: Dict[str, float] = {}
        for s, pid in zip(sims.tolist(), owner):
            if s > out.get(pid, -2.0):
                out[pid] = s
        return out

    # ------------------------------------------------------------------ write
    def enroll(self, engine, name: str, roll_no: str, role: str = "Student",
               department: str = "", photos: Optional[List[Tuple[str, bytes]]] = None,
               duplicate_thr: float = 0.55, person_id: Optional[str] = None) -> Tuple[dict, List[str]]:
        """Register a new person (person_id=None) or add photos to an existing one.
        photos = [(angle, jpeg/png bytes), ...]. Returns (person, notes)."""
        name = (name or "").strip()
        roll_no = (roll_no or "").strip()
        photos = [(a, b) for a, b in (photos or []) if b]
        if engine is None:
            raise EnrollError("Face models are not installed. Run: python download_models.py")
        if person_id is None:
            if not name:
                raise EnrollError("Please enter the person's name.")
            if not roll_no:
                raise EnrollError("Please enter a roll number / employee ID.")
            if not photos:
                raise EnrollError("Please add at least the FRONT photo (front + left + right is best).")
        elif not photos:
            raise EnrollError("No photos were given.")

        with self._lock:
            if person_id is None:
                for p in self.people.values():
                    if p.get("roll_no", "").lower() == roll_no.lower():
                        raise EnrollError(f"Roll number '{roll_no}' is already registered to {p['name']}.")
            elif person_id not in self.people:
                raise EnrollError("This person does not exist any more.")

        # ---- turn every photo into fingerprints (outside the lock: slow) ----
        notes, good = [], []     # good: (angle, raw bytes, sample)
        for angle, data in photos:
            label = angle if angle in ANGLES else "extra"
            img = decode_image(data)
            if img is None:
                notes.append(f"{label} photo: not a readable image, skipped.")
                continue
            sample = engine.extract_largest(img)
            if sample is None:
                notes.append(f"{label} photo: no clear face found, skipped. Use a well-lit photo "
                             f"where the face is not covered (side photos: turn the head ~30-45 degrees, "
                             f"not a full side view).")
                continue
            good.append((label, data, sample))
        if not good:
            raise EnrollError("No usable face was found in any photo. " + " ".join(notes))

        vecs = []
        for _, _, s in good:
            vecs.append(s.vec)
            if s.aligned is not None and hasattr(engine, "embed_aligned"):
                mv = engine.embed_aligned(cv2.flip(s.aligned, 1))        # mirrored copy
                if mv is not None:
                    vecs.append(mv)
        vecs = np.vstack(vecs).astype(np.float32)

        # ---- the photos must all be ONE human, and not someone else ----------
        if len(good) > 1:
            base = good[0][2].vec
            for label, _, s in good[1:]:
                if float(s.vec @ base) < 0.20:
                    raise EnrollError(f"The {label} photo does not look like the same person as the "
                                      f"{good[0][0]} photo. Please check the photos.")
        if person_id is not None:
            for label, _, s in good:
                own = self.scores(s.vec).get(person_id, -1.0)
                if own < 0.20:
                    raise EnrollError(f"The {label} photo does not look like the person already "
                                      f"registered here. Please check the photo.")
        for v in (s.vec for _, _, s in good):
            for other, score in self.scores(v).items():
                if other != person_id and score >= duplicate_thr:
                    o = self.people.get(other, {})
                    raise EnrollError(f"This face looks like {o.get('name', 'someone')} "
                                      f"({o.get('roll_no', '')}), who is already registered "
                                      f"(similarity {score:.2f}). Same person registered twice?")

        with self._lock:
            new = person_id is None
            pid = person_id or uuid.uuid4().hex[:8]
            folder = self._folder(pid)
            os.makedirs(folder, exist_ok=True)
            p = self.people.get(pid) or {
                "id": pid, "name": name, "roll_no": roll_no, "role": role or "Student",
                "department": department or "", "created": datetime.now().isoformat(timespec="seconds"),
                "photos": []}
            n0 = len(p["photos"])
            for i, (label, data, s) in enumerate(good, start=n0 + 1):
                with open(os.path.join(folder, f"photo_{i}.jpg"), "wb") as fh:
                    img = decode_image(data)
                    h, w = img.shape[:2]
                    k = self.max_px / max(h, w)
                    if k < 1:                                  # 4000px phone photo -> 800px
                        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality,
                                                         cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                    fh.write(buf.tobytes() if ok else data)
                if s.thumb_jpg:
                    with open(os.path.join(folder, f"face_{i}.jpg"), "wb") as fh:
                        fh.write(s.thumb_jpg)
                p["photos"].append({"file": f"photo_{i}.jpg", "face": f"face_{i}.jpg" if s.thumb_jpg else None,
                                    "angle": label, "quality": round(float(s.quality), 3)})
            old = self._emb.get(pid)
            E = vecs if old is None else np.vstack([old, vecs])
            np.save(os.path.join(folder, "emb.npy"), E)
            self._emb[pid] = E
            p["samples"] = int(len(E))
            p["angles"] = sorted({ph["angle"] for ph in p["photos"]})
            self.people[pid] = p
            self._save_index()
            self._rebuild()

        missing = [a for a in ("front", "left", "right") if a not in p["angles"]]
        if missing and new:
            notes.append("Tip: also add the " + " and ".join(missing) +
                         " photo(s) so the person is recognised from every angle.")
        return dict(p), notes

    def update_info(self, pid: str, **fields) -> dict:
        with self._lock:
            p = self.people.get(pid)
            if p is None:
                raise KeyError(pid)
            for k in ("name", "roll_no", "role", "department"):
                v = fields.get(k)
                if v is not None and str(v).strip():
                    p[k] = str(v).strip()
            self._save_index()
            return dict(p)

    def delete(self, pid: str) -> bool:
        with self._lock:
            if pid not in self.people:
                return False
            self.people.pop(pid, None)
            self._emb.pop(pid, None)
            shutil.rmtree(self._folder(pid), ignore_errors=True)
            self._save_index()
            self._rebuild()
            return True
