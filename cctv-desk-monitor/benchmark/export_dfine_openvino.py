#!/usr/bin/env python3
"""
export D-FINE จาก transformers เป็น OpenVINO IR (FP32 + INT8)

ทำไมต้องมีสคริปต์นี้ — ต่างจาก YOLO ที่ ultralytics มี `.export(format="openvino")`
ให้ในบรรทัดเดียว D-FINE เป็นโมเดล transformers ธรรมดาจึงต้องแปลงเอง 3 ขั้น:

  1. ห่อ model ให้คืนแค่ (logits, pred_boxes) — ตัด head อื่น 6 ตัวออกให้กราฟเล็กลง
  2. `ov.convert_model()` ด้วย static shape 1x3x640x640 (processor ของ D-FINE
     resize เป็น 640x640 คงที่อยู่แล้ว จึงไม่เสียอะไรจากการ fix shape
     และ OpenVINO optimize static shape ได้ดีกว่า dynamic)
  3. NNCF post-training quantization เป็น INT8

**ภาพ calibration ต้องไม่ทับกับชุดวัด** ไม่งั้นเป็น data leakage — สคริปต์นี้ใช้
ภาพใน `benchmark/images/` ที่ **ไม่ได้อยู่ใน `occlusion_testset.json`** (19 ใบที่ถูก
คัดออกตอนสร้างชุดทดสอบ ด้วยเกณฑ์เพดาน 60%) ซึ่งเป็น COCO val2017 โดเมนเดียวกัน
แต่ไม่ทับกับ 46 ใบที่ใช้วัด ถ้าไม่มีภาพเหล่านี้จะข้าม INT8 ไป

รัน:
    python benchmark/export_dfine_openvino.py               # dfine-n ทั้ง FP32 + INT8
    python benchmark/export_dfine_openvino.py dfine-s       # รุ่นอื่น
    python benchmark/export_dfine_openvino.py dfine-n --fp32-only

แล้ววัดผลด้วย:
    python benchmark/benchmark_models.py yolo11s.pt dfine-n dfine-n-ov dfine-n-ov-int8
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELDIR = ROOT / "models"
IMGDIR = HERE / "images"
META = HERE / "occlusion_testset.json"

IMGSZ = 640  # processor ของ D-FINE resize เป็น 640x640 คงที่

DFINE_IDS = {
    "dfine-n": "ustc-community/dfine-nano-coco",
    "dfine-s": "ustc-community/dfine-small-coco",
    "dfine-m": "ustc-community/dfine-medium-coco",
}


class LogitsAndBoxes(torch.nn.Module):
    """คืนแค่ 2 output ที่ postprocess ต้องใช้จริง"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def calibration_images() -> list[Path]:
    """ภาพ COCO ที่มีในเครื่องแต่ไม่ได้อยู่ในชุดวัด — กัน data leakage"""
    if not META.exists() or not IMGDIR.exists():
        return []
    used = {r["file"] for r in json.loads(META.read_text(encoding="utf-8"))}
    return sorted(p for p in IMGDIR.glob("*.jpg") if p.name not in used)


def make_preprocess(model_id: str):
    """ใช้ processor ตัวจริงของโมเดล ไม่เขียน preprocessing เอง

    เคยลองเขียนเองด้วย cv2 แล้วได้ภาพต่างจาก processor 1/255 (1 ระดับสี) จาก
    interpolation ที่ไม่เหมือนกัน — เล็กมากแต่ไม่มีเหตุผลให้รับความเสี่ยงนั้น
    เพราะ calibration ต้องเห็น distribution เดียวกับที่ inference จะเจอจริง
    """
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_id)

    def preprocess(path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return proc(images=rgb, return_tensors="np")["pixel_values"]

    return preprocess


def main(argv: list[str]) -> int:
    name = next((a for a in argv if not a.startswith("-")), "dfine-n")
    fp32_only = "--fp32-only" in argv
    if name not in DFINE_IDS:
        print(f"ไม่รู้จัก {name} — เลือกจาก: {', '.join(DFINE_IDS)}")
        return 1

    from transformers import AutoModelForObjectDetection

    print(f"โหลด {DFINE_IDS[name]} ...")
    torch_model = AutoModelForObjectDetection.from_pretrained(DFINE_IDS[name]).eval()

    print(f"แปลงเป็น OpenVINO IR (static {IMGSZ}x{IMGSZ}) ...")
    ov_model = ov.convert_model(
        LogitsAndBoxes(torch_model),
        example_input=torch.zeros(1, 3, IMGSZ, IMGSZ),
        input=[("pixel_values", [1, 3, IMGSZ, IMGSZ])],
    )

    fp32_dir = MODELDIR / f"{name}_openvino_model"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, fp32_dir / "model.xml", compress_to_fp16=False)
    print(f"  -> {fp32_dir.relative_to(ROOT)}")

    if fp32_only:
        return 0

    calib = calibration_images()
    if not calib:
        print("ไม่มีภาพ calibration ที่อยู่นอกชุดวัด — ข้าม INT8")
        print("(รัน benchmark_models.py ก่อนเพื่อโหลดภาพ COCO ลง benchmark/images/)")
        return 0

    import nncf

    print(f"\nquantize เป็น INT8 ด้วยภาพ calibration {len(calib)} ใบ "
          f"(นอกชุดวัด 46 ใบ) ...")
    preprocess = make_preprocess(DFINE_IDS[name])
    dataset = nncf.Dataset(calib, lambda p: {"pixel_values": preprocess(p)})
    int8_model = nncf.quantize(
        ov_model,
        dataset,
        subset_size=len(calib),
        # transformer ต้องใช้ preset นี้ ไม่งั้น attention/softmax พังแล้ว recall ตก
        model_type=nncf.ModelType.TRANSFORMER,
    )

    int8_dir = MODELDIR / f"{name}_int8_openvino_model"
    int8_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(int8_model, int8_dir / "model.xml")
    print(f"  -> {int8_dir.relative_to(ROOT)}")

    for d in (fp32_dir, int8_dir):
        mb = sum(f.stat().st_size for f in d.glob("*")) / 1e6
        print(f"  {d.name:<34} {mb:6.1f} MB")

    print(f"\nวัดผล:  python benchmark/benchmark_models.py "
          f"yolo11s.pt {name} {name}-ov {name}-ov-int8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
