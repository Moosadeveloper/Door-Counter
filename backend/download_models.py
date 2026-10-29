"""
download_models.py
------------------
Run ONCE before the first analysis:

    python download_models.py

Downloads the two free OpenCV face models into backend/models/
  * face_detection_yunet_2023mar.onnx    (finds faces,       ~0.2 MB)
  * face_recognition_sface_2021dec.onnx  (face fingerprint,  ~37 MB)

Source: https://github.com/opencv/opencv_zoo  (Apache-2.0 licence)
The YOLO weights (yolov8n.pt) are already inside models/.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

ZOO = "opencv/opencv_zoo"
FILES = {
    "face_detection_yunet_2023mar.onnx": ("face_detection_yunet", 100_000),
    "face_recognition_sface_2021dec.onnx": ("face_recognition_sface", 20_000_000),
}


def urls_for(name: str, folder: str):
    return [
        f"https://github.com/{ZOO}/raw/main/models/{folder}/{name}",
        f"https://media.githubusercontent.com/media/{ZOO}/main/models/{folder}/{name}",
    ]


def looks_valid(path: str, min_bytes: int) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
        return False
    with open(path, "rb") as fh:                       # a git-lfs "pointer" is a tiny text file
        return not fh.read(40).startswith(b"version https://git-lfs")


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    failed = []
    for name, (folder, min_bytes) in FILES.items():
        dest = os.path.join(MODELS_DIR, name)
        if looks_valid(dest, min_bytes):
            print(f"OK   {name} (already there)")
            continue
        print(f"GET  {name}")
        for url in urls_for(name, folder):
            try:
                download(url, dest)
                if looks_valid(dest, min_bytes):
                    print(f"OK   {name}")
                    break
                print("     downloaded file looks wrong, trying the next address")
            except Exception as e:
                print(f"     failed: {e}")
        else:
            failed.append((name, folder))
            if os.path.exists(dest):
                os.remove(dest)

    if failed:
        print("\nCould not download everything. Do it by hand:")
        for name, folder in failed:
            print(f"  open  https://github.com/{ZOO}/blob/main/models/{folder}/{name}")
            print(f"  click the download icon and save the file into:  {MODELS_DIR}")
        return 1

    try:                                              # final check: can OpenCV load them?
        import cv2
        from faces import FaceEngine
        FaceEngine(MODELS_DIR)
        print("\nFace models load correctly (OpenCV", cv2.__version__ + "). You are ready.")
    except Exception as e:
        print(f"\nFiles are there but could not be loaded: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
