import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager

cfg = load_config(str(ROOT / "configs" / "default.yaml"))

d = DetectorManager(cfg)

for k, v in d.detectors.items():
    obj = v.get("obj")
    if v.get("type") == "yolo" and hasattr(obj, "model_name"):
        print(k, "model_name=", obj.model_name)
    else:
        print(k, "type=", v.get("type"))
