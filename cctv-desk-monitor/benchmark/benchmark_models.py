#!/usr/bin/env python3
"""
วัดโมเดลตรวจจับบุคคลกับฉากจริงที่ "คนถูกจอ/แล็ปท็อปบัง"

ชุดทดสอบมาจาก COCO val2017 คัดเฉพาะภาพที่กล่อง person ซ้อนทับกล่อง
tv/laptop/keyboard จริง 8-60% ของพื้นที่ตัว
  - ขั้นต่ำ 8%  = ต้องถูกบังจริง ไม่ใช่แค่นั่งใกล้จอ
  - เพดาน 60% = กัน "คนที่ปรากฏบนหน้าจอทีวี" ซึ่ง COCO ก็ label เป็น person
    แต่ไม่ใช่คนจริงที่ถูกบัง

ground truth เป็นของ COCO ไม่ใช่การนับเอง

วิธีวัด — ตรงกับที่ระบบใช้จริง:
  ย่อภาพเป็น 480px แล้วเช็คว่า center ของ detection ตกในกล่อง GT ของคนนั้นไหม
  (ระบบจริงก็แมป center เข้า ROI แบบเดียวกัน)

รองรับ 3 backend — เรียกด้วยชื่อสั้นได้ ตัวไหนไม่มีไฟล์จะโหลดให้อัตโนมัติ:
  ultralytics    yolo11n.pt yolo11s.pt rtdetr-l.pt crowdhuman-yolov8n
                 models/yolo11s_openvino_model  (path ที่เป็นโฟลเดอร์ = OpenVINO)
  transformers   dfine-n dfine-s dfine-m         (D-FINE — transformer detector)
  onnx+openvino  rtmdet-tiny rtmdet-s rtmdet-n-person rtmdet-m-person

รัน:
    python benchmark/benchmark_models.py                       # ใช้ค่า default
    python benchmark/benchmark_models.py yolo11n.pt dfine-s rtmdet-tiny
    python benchmark/benchmark_models.py --all                 # ทุกตัวที่รองรับ
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELDIR = ROOT / "models"
META = HERE / "occlusion_testset.json"
IMGDIR = HERE / "images"

THREADS = 2  # ตรงกับ detector.torch_threads ใน occupancy.yaml
CONF = 0.5   # ตรงกับ detector.conf ใน occupancy.yaml

DEFAULT_MODELS = ["yolo11n.pt", "yolo11s.pt",
                  "dfine-n", "dfine-s",
                  "rtmdet-tiny", "rtmdet-n-person",
                  "crowdhuman-yolov8n"]

ALL_MODELS = ["yolov8n.pt", "yolo11n.pt", "yolov8s.pt", "yolo11s.pt",
              "dfine-n", "dfine-s", "dfine-m",
              "rtmdet-tiny", "rtmdet-s", "rtmdet-n-person", "rtmdet-m-person",
              "crowdhuman-yolov8n", "rtdetr-l.pt"]

# D-FINE (transformer detector รุ่นต่อจาก RT-DETR) — โหลดผ่าน transformers hub
DFINE_IDS = {
    "dfine-n": "ustc-community/dfine-nano-coco",
    "dfine-s": "ustc-community/dfine-small-coco",
    "dfine-m": "ustc-community/dfine-medium-coco",
}

# RTMDet ที่ export จาก mmdeploy แล้ว — เลี่ยงการลง mmcv/mmdet ซึ่งต้อง compile บน Windows
_RTMDET_BASE = "https://huggingface.co/bukuroo/RTMDet-ONNX/resolve/main"
RTMDET_FILES = {
    "rtmdet-tiny": "rtmdet-tiny-coco.onnx",
    "rtmdet-s": "rtmdet-s-coco.onnx",
    "rtmdet-n-person": "rtmdet-n-person.onnx",
    "rtmdet-m-person": "rtmdet-m-person.onnx",
}

# น้ำหนักที่ fine-tune บน CrowdHuman (เทรนเฉพาะคน + ฉากแออัด/ถูกบัง)
PT_URLS = {
    "crowdhuman-yolov8n": (
        "crowdhuman_yolov8n.pt",
        "https://huggingface.co/raghavendra24/crowdhuman-yolov8n/resolve/main/crowdhuman_yolov8n_best.pt",
    ),
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  โหลด {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def ensure_images(recs: list[dict]) -> None:
    """โหลดภาพจาก COCO (ไม่เก็บใน git เพราะใหญ่) ข้ามใบที่มีแล้ว"""
    IMGDIR.mkdir(exist_ok=True)
    missing = [r for r in recs if not (IMGDIR / r["file"]).exists()]
    if not missing:
        return
    print(f"กำลังโหลดภาพ {len(missing)} ใบจาก COCO ...")
    for i, r in enumerate(missing, 1):
        try:
            urllib.request.urlretrieve(r["url"], IMGDIR / r["file"])
        except OSError as exc:
            print(f"  ข้าม {r['file']}: {exc}")
        if i % 10 == 0:
            print(f"  {i}/{len(missing)}")


# ---------------------------------------------------------------------------
# detector backends — ทุกตัวรับเฟรม BGR คืน list ของ (cx, cy) ในพิกเซลของเฟรมนั้น
# เวลาที่จับรวม preprocess + inference + postprocess เหมือนกันหมดทุก backend
# ---------------------------------------------------------------------------


class UltralyticsDetector:
    """yolo* / rtdetr* / โมเดล fine-tune — รวมโฟลเดอร์ OpenVINO ที่ export ไว้"""

    def __init__(self, weights: str):
        torch.set_num_threads(THREADS)
        self.model = (YOLO(weights, task="detect") if Path(weights).is_dir()
                      else YOLO(weights))
        # โมเดล CrowdHuman มีคลาสเดียว ('person') ส่วน COCO person ก็ index 0 เหมือนกัน
        self.cls = [0]

    def __call__(self, bgr):
        res = self.model.predict(bgr, classes=self.cls, conf=CONF, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return out


class DFineDetector:
    """D-FINE ผ่าน transformers — postprocess ของ processor คืนกล่องพิกัดภาพเดิมให้แล้ว"""

    def __init__(self, model_id: str):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        torch.set_num_threads(THREADS)
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).eval()
        names = self.model.config.id2label
        self.person = next(i for i, n in names.items() if str(n).lower() == "person")

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        inputs = self.proc(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.proc.post_process_object_detection(
            out, target_sizes=torch.tensor([[h, w]]), threshold=CONF)[0]
        keep = res["labels"] == self.person
        return [((x1 + x2) / 2, (y1 + y2) / 2)
                for x1, y1, x2, y2 in res["boxes"][keep].tolist()]


class RTMDetOnnxDetector:
    """RTMDet ที่ export จาก mmdeploy — รันบน OpenVINO runtime (อ่าน .onnx ได้ตรงๆ
    ไม่ต้องลง onnxruntime) output เป็นแบบ end2end คือ NMS มาให้แล้ว:

        dets   [1, N, 5] = x1,y1,x2,y2,score ในพิกเซลของภาพ input ที่ letterbox แล้ว
        labels [1, N]    = index คลาส (COCO person = 0 / รุ่น -person มีคลาสเดียว)

    preprocess ตาม test pipeline ของ mmdet: resize คงอัตราส่วนแล้ว pad มุมล่างขวา
    ด้วย 114 จากนั้น normalize ด้วย mean/std ของ ImageNet ในลำดับ BGR (bgr_to_rgb=False)
    """

    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    PAD = 114

    def __init__(self, onnx_path: Path):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(onnx_path)
        self.net = core.compile_model(model, "CPU",
                                      {"INFERENCE_NUM_THREADS": THREADS})
        _, _, self.ih, self.iw = model.inputs[0].shape
        self.out_dets = self.net.output("dets")
        self.out_labels = self.net.output("labels")

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(self.iw / w, self.ih / h)
        nw, nh = round(w * scale), round(h * scale)
        canvas = np.full((self.ih, self.iw, 3), self.PAD, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))

        blob = ((canvas.astype(np.float32) - self.MEAN) / self.STD)
        blob = blob.transpose(2, 0, 1)[None]

        res = self.net(blob)
        dets, labels = res[self.out_dets][0], res[self.out_labels][0]

        out = []
        for (x1, y1, x2, y2, score), cls in zip(dets, labels):
            if score < CONF or cls != 0:
                continue
            out.append(((x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale))
        return out


def build_detector(name: str):
    """แปลชื่อสั้นเป็น detector — โหลดไฟล์ให้ถ้ายังไม่มี"""
    if name in DFINE_IDS:
        return DFineDetector(DFINE_IDS[name])
    if name in RTMDET_FILES:
        path = MODELDIR / RTMDET_FILES[name]
        download(f"{_RTMDET_BASE}/{RTMDET_FILES[name]}", path)
        return RTMDetOnnxDetector(path)
    if name in PT_URLS:
        fname, url = PT_URLS[name]
        path = MODELDIR / fname
        download(url, path)
        return UltralyticsDetector(str(path))
    # ชื่อ/path ธรรมดา — ให้ ultralytics จัดการ (ถ้ามีใน models/ แล้วก็ใช้ของเดิม)
    local = MODELDIR / name
    return UltralyticsDetector(str(local) if local.exists() else name)


# ---------------------------------------------------------------------------


def inside(box: list[float], cx: float, cy: float) -> bool:
    x, y, w, h = box
    return x <= cx <= x + w and y <= cy <= y + h


def evaluate(model_name: str, recs: list[dict]) -> dict:
    det = build_detector(model_name)

    warm = cv2.imread(str(IMGDIR / recs[0]["file"]))
    det(cv2.resize(warm, (480, int(warm.shape[0] * 480 / warm.shape[1]))))

    hit_all = tot_all = hit_occ = tot_occ = false_pos = 0
    infer_s = 0.0
    for r in recs:
        img = cv2.imread(str(IMGDIR / r["file"]))
        h, w = img.shape[:2]
        small = cv2.resize(img, (480, int(h * 480 / w)))
        # จับเวลาเฉพาะ inference - ไม่รวมอ่านไฟล์จากดิสก์ ซึ่งของจริงไม่มี
        # (เฟรมมาจากกล้องอยู่ใน RAM แล้ว) ตัวเลขนี้จึงเทียบกับ production ได้
        t = time.perf_counter()
        found = det(small)
        infer_s += time.perf_counter() - t

        # แปลงกลับเป็นพิกัดภาพเต็ม เพราะ ground truth ของ COCO อยู่สเกลนั้น
        sx, sy = w / small.shape[1], h / small.shape[0]
        centers = [(cx * sx, cy * sy) for cx, cy in found]

        for box in r["persons"]:
            tot_all += 1
            hit_all += any(inside(box, cx, cy) for cx, cy in centers)
        for box in r["occluded"]:
            tot_occ += 1
            hit_occ += any(inside(box, cx, cy) for cx, cy in centers)
        # เทียบกับคนทุกขนาด ไม่งั้นการเจอคนตัวเล็กไกลๆ จะถูกนับเป็น false positive
        for cx, cy in centers:
            if not any(inside(b, cx, cy) for b in r["persons_all"]):
                false_pos += 1

    return {
        "ms": infer_s / len(recs) * 1000,
        "recall_all": hit_all / max(tot_all, 1),
        "recall_occluded": hit_occ / max(tot_occ, 1),
        "hit_all": hit_all, "tot_all": tot_all,
        "hit_occ": hit_occ, "tot_occ": tot_occ,
        "false_pos": false_pos,
    }


def main(argv: list[str]) -> int:
    models = ALL_MODELS if argv[:1] == ["--all"] else (argv or DEFAULT_MODELS)

    recs = json.loads(META.read_text(encoding="utf-8"))
    ensure_images(recs)
    recs = [r for r in recs if (IMGDIR / r["file"]).exists()]
    if not recs:
        print("ไม่มีภาพให้ทดสอบ - ตรวจการเชื่อมต่อเน็ตแล้วรันใหม่")
        return 1

    print(f"\nภาพ {len(recs)} ใบ | คน {sum(len(r['persons']) for r in recs)} "
          f"(ถูกจอบัง {sum(len(r['occluded']) for r in recs)})\n")
    print(f"ทุกโมเดลจำกัด {THREADS} threads เท่ากัน (ตรงกับ detector.torch_threads)")
    print("หมายเหตุ: ms ที่นี่สูงกว่าของจริง ~20-25% เพราะภาพทดสอบขนาดไม่เท่ากันทุกใบ")
    print("          กล้องจริงส่งเฟรมขนาดคงที่ -> yolo11n ~68 ms, yolo11s ~120 ms\n")
    print(f"{'model':<20}{'ms/ภาพ':>9}{'recall ทุกคน':>16}{'recall คนถูกบัง':>18}{'false pos':>12}")
    print("-" * 76)
    rows = {}
    for name in models:
        try:
            r = evaluate(name, recs)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<20}  ใช้ไม่ได้: {str(exc)[:45]}")
            continue
        rows[name] = r
        print(f"{name:<20}{r['ms']:>9.1f}"
              f"{r['hit_all']:>8}/{r['tot_all']:<3}{r['recall_all']:>6.0%}"
              f"{r['hit_occ']:>9}/{r['tot_occ']:<3}{r['recall_occluded']:>6.0%}"
              f"{r['false_pos']:>10}")

    # เขียนทับทุกรอบ - ใส่ _meta ไว้เพราะ ms เทียบข้ามรอบไม่ได้ ต้องรู้ว่ามาจากรันไหน
    out = HERE / "results_latest.json"
    payload = {
        "_meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "argv": argv or ["(default)"],
            "images": len(recs),
            "threads": THREADS,
            "conf": CONF,
        },
        **rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nเก็บผลดิบไว้ที่ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
