"""Quick local test to verify YOLO models and camera pipeline.

It will:
- Open webcam index 0
- Capture a single frame
- Attempt to load models: yolov8n.pt and yolo26n-face.pt (if present)
- Run inference and print detections
- Save annotated image to ./files/test_out.jpg
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2

models = [
    ROOT / "models" / "yolov8n.pt",
    ROOT / "models" / "yolo26n-face.pt",
]
loaded = {}

print("Python", sys.version)
try:
    import ultralytics

    print("ultralytics", ultralytics.__version__)
except Exception as e:
    print("ultralytics import failed:", e)

for model_path in models:
    p = Path(model_path)
    if not p.exists():
        print(f"Model {p} not found in CWD")
        continue
    try:
        from ultralytics import YOLO

        print(f"Loading model {p}")
        mdl = YOLO(str(p))
        loaded[str(p.name)] = mdl
        print(f"Loaded {p}")
    except Exception as e:
        print(f"Failed to load {p}:", e)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera 0")
    sys.exit(1)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to read frame from camera")
    sys.exit(1)

h, w = frame.shape[:2]
print("Captured frame", w, "x", h)

out_img = frame.copy()

for name, mdl in loaded.items():
    print(f"Running inference with {name}...")
    try:
        t0 = time.time()
        results = mdl.predict(frame)
        t1 = time.time()
        print(f"Inference time: {(t1 - t0) * 1000:.1f} ms")
        for r in results:
            boxes = getattr(r, "boxes", [])
            for b in boxes:
                try:
                    xyxy = (
                        b.xyxy[0].cpu().numpy()
                        if hasattr(b.xyxy[0], "cpu")
                        else b.xyxy[0]
                    )
                    conf = (
                        float(b.conf[0])
                        if hasattr(b, "conf")
                        else float(b.conf)
                    )
                    cls = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
                    x1, y1, x2, y2 = map(int, xyxy)
                    print(
                        f" {name}: cls={cls} conf={conf:.2f} bbox={[x1, y1, x2, y2]}"
                    )
                    cv2.rectangle(
                        out_img, (x1, y1), (x2, y2), (10, 200, 10), 2
                    )
                    cv2.putText(
                        out_img,
                        f"{name} {conf:.2f}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                except Exception as e:
                    print(" box parse error:", e)
    except Exception as e:
        print(f"Inference failed for {name}:", e)

out_dir = ROOT / "files"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"inference_{int(time.time())}.jpg"
cv2.imwrite(str(out_path), out_img)
print("Wrote", out_path)
print("Done")
