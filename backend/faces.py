"""
faces.py
--------
Free, lightweight face recognition using ONLY OpenCV:
  * YuNet  - finds faces          (~0.3 MB ONNX file)
  * SFace  - turns a face into a 128-number "fingerprint" (~37 MB ONNX file)

Two faces of the same person give fingerprints with high cosine similarity
(> ~0.4); different people give low similarity.
Run `python download_models.py` once to fetch the two model files.
"""
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"


@dataclass
class FaceSample:
    vec: np.ndarray                 # L2-normalised embedding (128,)
    quality: float                  # 0..1, bigger = better looking sample
    thumb_jpg: Optional[bytes] = None
    aligned: Optional[np.ndarray] = None     # 112x112 aligned face (for mirrored samples)


class FaceModelsMissing(RuntimeError):
    pass


def models_available(models_dir: str) -> bool:
    return (os.path.isfile(os.path.join(models_dir, YUNET_FILE))
            and os.path.isfile(os.path.join(models_dir, SFACE_FILE))
            and hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"))


class FaceEngine:
    name = "opencv-yunet-sface"

    def __init__(self, models_dir: str, det_score: float = 0.7, min_face_px: int = 24):
        if not models_available(models_dir):
            raise FaceModelsMissing(
                "Face models not found (or OpenCV too old, need >= 4.8). "
                "Run: python download_models.py")
        self.det_score = det_score
        self.min_face_px = min_face_px
        self._det = cv2.FaceDetectorYN.create(
            os.path.join(models_dir, YUNET_FILE), "", (320, 320), 0.6, 0.3, 50)
        self._rec = cv2.FaceRecognizerSF.create(
            os.path.join(models_dir, SFACE_FILE), "")
        # OpenCV's DNN nets are not safe to call from two threads at once
        # (live camera thread + a registration request), so calls are serialised.
        self._lock = threading.Lock()

    def extract(self, frame: np.ndarray, box: Tuple[float, float, float, float],
                meta: Optional[dict] = None) -> Optional[FaceSample]:
        """Look for a face in the head area of one person box."""
        with self._lock:
            return self._extract(frame, box)

    def _extract(self, frame, box) -> Optional[FaceSample]:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 20:
            return None
        pad = int(0.10 * bw)
        cx1, cx2 = max(0, x1 - pad), min(W, x2 + pad)
        cy1, cy2 = max(0, y1), min(H, y1 + int(0.5 * bh))     # head + shoulders
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        scale = 1.0
        longest = max(crop.shape[:2])
        if longest < 160:                                       # tiny crop: enlarge first
            scale = min(4.0, 160.0 / longest)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        self._det.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self._det.detect(crop)
        if faces is None or len(faces) == 0:
            return None

        best = max(faces, key=lambda f: float(f[14]) * float(f[2]) * float(f[3]))
        score, face_w = float(best[14]), float(best[2]) / scale
        if score < self.det_score or face_w < self.min_face_px:
            return None

        return self._embed(crop, best, score, face_w)

    def extract_largest(self, image: np.ndarray, min_score: float = 0.6) -> Optional[FaceSample]:
        """Registration photos: find the BIGGEST face in a whole photo (not a
        person box). A slightly lower detector score is accepted because side
        (left / right) photos give lower scores than a straight face."""
        if image is None or image.size == 0:
            return None
        img, scale = image, 1.0
        longest = max(img.shape[:2])
        if longest > 1280:                                       # phone photos: shrink
            scale = 1280.0 / longest
        elif longest < 320:                                      # tiny photos: enlarge
            scale = 320.0 / longest
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        with self._lock:
            self._det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self._det.detect(img)
            if faces is None or len(faces) == 0:
                return None
            best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
            score, face_w = float(best[14]), float(best[2]) / scale
            if score < min_score or face_w < self.min_face_px:
                return None
            return self._embed(img, best, score, face_w)

    def embed_aligned(self, aligned: np.ndarray) -> Optional[np.ndarray]:
        """Fingerprint of an already aligned 112x112 face (used for mirrored copies)."""
        with self._lock:
            vec = self._rec.feature(aligned).flatten().astype(np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n > 1e-6 else None

    def _embed(self, img: np.ndarray, face_row, score: float, face_w: float) -> Optional[FaceSample]:
        aligned = self._rec.alignCrop(img, face_row)          # upright, centred, eyes-level face
        vec = self._rec.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        vec /= norm

        thumb = None
        if aligned.size:
            thumb = _clean_face_thumb(aligned)

        # Penalise blurry/motion-smeared faces (common when someone walks quickly
        # through a doorway) so a sharp frame wins the "best face" slot even if a
        # blurry one had a slightly higher detector score.
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_score = min(1.0, sharpness / 250.0)
        quality = score * min(1.0, face_w / 64.0) * (0.35 + 0.65 * sharp_score)

        return FaceSample(vec=vec, quality=quality, thumb_jpg=thumb, aligned=aligned)


def _clean_face_thumb(aligned: np.ndarray, out_px: int = 160) -> Optional[bytes]:
    """Turn the raw aligned face crop into a clear thumbnail for the results page:
    upscale, denoise, sharpen and lift contrast a little (CCTV faces are usually
    small, soft and dim). Uses the SAME aligned crop SFace itself compares against,
    so what the user sees is exactly what is being matched - no stray shoulders or
    background like a plain padded box crop would include."""
    th = cv2.resize(aligned, (out_px, out_px), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(th, d=5, sigmaColor=40, sigmaSpace=40)   # smooth JPEG/CCTV noise, keep edges
    blur = cv2.GaussianBlur(denoised, (0, 0), 2.2)
    sharp = cv2.addWeighted(denoised, 1.6, blur, -0.6, 0)                  # unsharp mask
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)       # local contrast, dark footage
    final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    ok, buf = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None
