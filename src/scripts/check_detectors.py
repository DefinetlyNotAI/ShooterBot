import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference import DetectorManager

cfg = load_config(str(ROOT / "configs" / "default.yaml"))

d = DetectorManager(cfg)

print("detector keys:", list(d.detectors.keys()))

for k, v in d.detectors.items():
    print(k, "type=", v.get("type"), "priority=", v.get("priority"))
    obj = v.get("obj")
    try:
        print("  obj repr:", repr(obj)[:200])
    except Exception:
        pass

print("device:", d.device)
