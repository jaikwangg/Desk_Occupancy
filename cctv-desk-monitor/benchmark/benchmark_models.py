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

รองรับ 4 backend — เรียกด้วยชื่อสั้นได้ ตัวไหนไม่มีไฟล์จะโหลดให้อัตโนมัติ:
  ultralytics    yolo11n.pt yolo11s.pt rtdetr-l.pt crowdhuman-yolov8n
                 models/yolo11s_openvino_model  (path ที่เป็นโฟลเดอร์ = OpenVINO)
  transformers   dfine-n dfine-s dfine-m         (D-FINE — transformer detector)
  dfine+openvino dfine-n-ov dfine-n-ov-int8     (ต้อง export ก่อน ดู export_dfine_openvino.py)
  onnx+openvino  rtmdet-tiny rtmdet-s rtmdet-n-person rtmdet-m-person

รัน:
    python benchmark/benchmark_models.py                       # ใช้ค่า default
    python benchmark/benchmark_models.py yolo11n.pt dfine-s rtmdet-tiny
    python benchmark/benchmark_models.py --all                 # ทุกตัวที่รองรับ
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELDIR = ROOT / "models"
TESTSETS = {"occlusion": HERE / "occlusion_testset.json",
            "desk": HERE / "desk_testset.json",
            "openimages": HERE / "openimages_testset.json"}

# คำเตือนที่ต้องขึ้นก่อนตารางของบางชุด — ตัวเลขบางคอลัมน์เชื่อไม่ได้
TESTSET_NOTES = {
    "openimages": (
        "*** ชุดนี้อ่านได้เฉพาะคอลัมน์ recall ***",
        "    คอลัมน์ false pos ใช้ไม่ได้ เพราะกล่องคนของ Open Images ไม่ครบ",
        "    (ตรวจแล้ว: FP เกือบทั้งหมดตกบนคนจริงที่ไม่มีกล่อง GT)",
        "    และชุดนี้ไม่มีภาพโต๊ะว่าง — ใช้ --testset desk สำหรับ false PRESENT",
    ),
}
IMGDIR = HERE / "images"

THREADS = 2  # ตรงกับ detector.torch_threads ใน occupancy.yaml
CONF = 0.5   # ตรงกับ detector.conf ใน occupancy.yaml

DEFAULT_MODELS = ["yolo11n.pt", "yolo11s.pt",
                  "dfine-n", "dfine-s",
                  "rtmdet-tiny", "rtmdet-n-person",
                  "crowdhuman-yolov8n"]

ALL_MODELS = ["yolov8n.pt", "yolo11n.pt", "yolov8s.pt", "yolo11s.pt",
              "dfine-n", "dfine-s", "dfine-m",
              "dfine-n-ov", "dfine-n-ov-int8",
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
    """โหลดภาพจาก URL ในชุดทดสอบ (COCO หรือ Open Images แล้วแต่ชุด)
    ไม่เก็บใน git เพราะใหญ่ — ข้ามใบที่มีแล้ว

    โหลดขนาน 16 เส้น เพราะเป็นงานรอเน็ตล้วน ไม่กิน CPU — ชุด openimages
    มี ~500 ใบ ถ้าโหลดทีละใบจะใช้เวลาเป็นสิบนาที
    """
    IMGDIR.mkdir(exist_ok=True)
    missing = [r for r in recs if not (IMGDIR / r["file"]).exists()]
    if not missing:
        return
    print(f"กำลังโหลดภาพ {len(missing)} ใบ ...")
    done = 0

    def fetch(r):
        try:
            # เขียนลงไฟล์ชั่วคราวก่อนแล้วค่อย rename — กัน error กลางคันทิ้งไฟล์
            # ที่โหลดไม่ครบไว้ ซึ่งรอบถัดไปจะนึกว่ามีแล้วและอ่านไม่ออก
            tmp = IMGDIR / (r["file"] + ".part")
            urllib.request.urlretrieve(r["url"], tmp)
            tmp.replace(IMGDIR / r["file"])
        except OSError as exc:
            return f"  ข้าม {r['file']}: {exc}"
        return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        for msg in pool.map(fetch, missing):
            if msg:
                print(msg)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(missing)}", flush=True)


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
    """D-FINE — postprocess ของ processor คืนกล่องพิกัดภาพเดิมให้แล้ว

    `ov_dir` = โฟลเดอร์ IR ที่ export ด้วย `benchmark/export_dfine_openvino.py`
    ถ้าใส่มาจะรัน forward บน OpenVINO แทน PyTorch โดย **pre/postprocess ใช้ของ
    เดิมทุกบรรทัด** — ต่างกันแค่ runtime ผลจึงเทียบกันได้ตรงๆ
    """

    def __init__(self, model_id: str, ov_dir: Path | None = None):
        from transformers import AutoImageProcessor, AutoConfig

        torch.set_num_threads(THREADS)
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        names = AutoConfig.from_pretrained(model_id).id2label
        self.person = next(i for i, n in names.items() if str(n).lower() == "person")

        self.net = None
        if ov_dir is None:
            from transformers import AutoModelForObjectDetection

            self.model = AutoModelForObjectDetection.from_pretrained(model_id).eval()
        else:
            import openvino as ov

            core = ov.Core()
            self.net = core.compile_model(ov_dir / "model.xml", "CPU",
                                          {"INFERENCE_NUM_THREADS": THREADS})

    def _forward(self, pixel_values):
        """คืน object ที่มี .logits/.pred_boxes ให้ post_process_object_detection ใช้"""
        if self.net is None:
            with torch.inference_mode():
                return self.model(pixel_values=pixel_values)
        logits, boxes = self.net(pixel_values.numpy()).to_tuple()
        return SimpleNamespace(logits=torch.from_numpy(logits),
                               pred_boxes=torch.from_numpy(boxes))

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        inputs = self.proc(images=rgb, return_tensors="pt")
        out = self._forward(inputs["pixel_values"])
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
    # dfine-n-ov / dfine-n-ov-int8 -> IR ที่ export ไว้ใน models/
    for suffix, folder in (("-ov-int8", "_int8_openvino_model"),
                           ("-ov", "_openvino_model")):
        if name.endswith(suffix) and name[:-len(suffix)] in DFINE_IDS:
            base = name[: -len(suffix)]
            ov_dir = MODELDIR / f"{base}{folder}"
            if not (ov_dir / "model.xml").exists():
                raise FileNotFoundError(
                    f"ยังไม่มี {ov_dir.name} — รัน "
                    f"python benchmark/export_dfine_openvino.py {base}")
            return DFineDetector(DFINE_IDS[base], ov_dir=ov_dir)
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
    neg_imgs = neg_fp = neg_hit = 0
    infer_s = 0.0
    for r in recs:
        img = cv2.imread(str(IMGDIR / r["file"]))
        if img is None:
            # ไฟล์โหลดมาไม่ครบ (เช่นถูก kill กลางทาง) — บอกชื่อไฟล์ไปเลย
            # ไม่งั้นจะได้แค่ 'NoneType' has no attribute 'shape' ซึ่งหาต้นเหตุยาก
            raise RuntimeError(
                f"อ่านภาพไม่ได้: {r['file']} — ลบไฟล์นี้แล้วรันใหม่ให้โหลดซ้ำ")
        h, w = img.shape[:2]
        # Open Images เก็บกล่องเป็นสัดส่วน 0-1 (CSV ไม่มีขนาดภาพ) ต้องคูณกลับ
        # ก่อนใช้ ส่วน COCO เก็บเป็นพิกเซลอยู่แล้ว
        scale = (lambda bs: [[bx * w, by * h, bw * w, bh * h] for bx, by, bw, bh in bs])             if r.get("norm") else (lambda bs: bs)
        persons, occluded = scale(r["persons"]), scale(r["occluded"])
        persons_all = scale(r["persons_all"])
        small = cv2.resize(img, (480, int(h * 480 / w)))
        # จับเวลาเฉพาะ inference - ไม่รวมอ่านไฟล์จากดิสก์ ซึ่งของจริงไม่มี
        # (เฟรมมาจากกล้องอยู่ใน RAM แล้ว) ตัวเลขนี้จึงเทียบกับ production ได้
        t = time.perf_counter()
        found = det(small)
        infer_s += time.perf_counter() - t

        # แปลงกลับเป็นพิกัดภาพเต็ม เพราะ ground truth ของ COCO อยู่สเกลนั้น
        sx, sy = w / small.shape[1], h / small.shape[0]
        centers = [(cx * sx, cy * sy) for cx, cy in found]

        for box in persons:
            tot_all += 1
            hit_all += any(inside(box, cx, cy) for cx, cy in centers)
        for box in occluded:
            tot_occ += 1
            hit_occ += any(inside(box, cx, cy) for cx, cy in centers)
        # เทียบกับคนทุกขนาด ไม่งั้นการเจอคนตัวเล็กไกลๆ จะถูกนับเป็น false positive
        for cx, cy in centers:
            if not any(inside(b, cx, cy) for b in persons_all):
                false_pos += 1
        # ภาพ negative = โต๊ะว่างไม่มีคนเลย ทุก detection คือ false PRESENT
        if r.get("kind") == "negative":
            neg_imgs += 1
            neg_fp += len(centers)
            neg_hit += bool(centers)

    return {
        "ms": infer_s / len(recs) * 1000,
        "recall_all": hit_all / max(tot_all, 1),
        "recall_occluded": hit_occ / max(tot_occ, 1),
        "hit_all": hit_all, "tot_all": tot_all,
        "hit_occ": hit_occ, "tot_occ": tot_occ,
        "false_pos": false_pos,
        # ภาพโต๊ะว่าง: neg_hit = จำนวนภาพที่เจอคนทั้งที่ไม่มีคน = false PRESENT
        "neg_imgs": neg_imgs, "neg_fp": neg_fp, "neg_hit": neg_hit,
    }


def main(argv: list[str]) -> int:
    which = "occlusion"
    for i, a in enumerate(argv):
        if a == "--testset" and i + 1 < len(argv):
            which = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
            break
    if which not in TESTSETS:
        print(f"ไม่รู้จักชุดทดสอบ {which} — เลือกจาก: {', '.join(TESTSETS)}")
        return 1
    META = TESTSETS[which]
    if not META.exists():
        print(f"ไม่มี {META.name} — รัน python benchmark/build_desk_testset.py ก่อน")
        return 1

    models = ALL_MODELS if argv[:1] == ["--all"] else (argv or DEFAULT_MODELS)

    recs = json.loads(META.read_text(encoding="utf-8"))
    ensure_images(recs)
    recs = [r for r in recs if (IMGDIR / r["file"]).exists()]
    if not recs:
        print("ไม่มีภาพให้ทดสอบ - ตรวจการเชื่อมต่อเน็ตแล้วรันใหม่")
        return 1

    print(f"\nภาพ {len(recs)} ใบ | คน {sum(len(r['persons']) for r in recs)} "
          f"(ถูกจอบัง {sum(len(r['occluded']) for r in recs)})\n")
    for note in TESTSET_NOTES.get(which, ()):
        print(note)
    if which in TESTSET_NOTES:
        print()
    print(f"ทุกโมเดลจำกัด {THREADS} threads เท่ากัน (ตรงกับ detector.torch_threads)")
    print("หมายเหตุ: ms ที่นี่สูงกว่าของจริง ~20-25% เพราะภาพทดสอบขนาดไม่เท่ากันทุกใบ")
    print("          กล้องจริงส่งเฟรมขนาดคงที่ -> yolo11n ~68 ms, yolo11s ~120 ms\n")
    print(f"{'model':<20}{'ms/ภาพ':>9}{'recall ทุกคน':>16}{'recall คนถูกบัง':>18}"
          f"{'false pos':>12}{'โต๊ะว่างพลาด':>16}")
    print("-" * 92)
    rows = {}
    for name in models:
        try:
            r = evaluate(name, recs)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<20}  ใช้ไม่ได้: {str(exc)[:45]}")
            continue
        finally:
            # คืนหน่วยความจำก่อนโหลดโมเดลตัวถัดไป — รันหลายตัวติดกันบนเครื่อง
            # แรมน้อยเคยโดน OOM kill กลางตาราง (runtime ของ OpenVINO/torch
            # ไม่ได้คืนทันทีตอน detector หลุด scope)
            gc.collect()
        rows[name] = r
        neg = (f"{r['neg_hit']}/{r['neg_imgs']}" if r["neg_imgs"] else "-")
        print(f"{name:<20}{r['ms']:>9.1f}"
              f"{r['hit_all']:>8}/{r['tot_all']:<3}{r['recall_all']:>6.0%}"
              f"{r['hit_occ']:>9}/{r['tot_occ']:<3}{r['recall_occluded']:>6.0%}"
              f"{r['false_pos']:>10}{neg:>14}")

    # เก็บ 2 ที่: results_latest.json (เขียนทับ) + results/ (เก็บถาวรไม่ทับกัน)
    # เพราะ ms เทียบข้ามรอบไม่ได้ ถ้าเขียนทับที่เดียวจะเสียผลรอบที่เอกสารอ้างถึง
    stamp = time.strftime("%Y-%m-%d_%H%M")
    payload = {
        "_meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "testset": which,
            "argv": argv or ["(default)"],
            "images": len(recs),
            "threads": THREADS,
            "conf": CONF,
        },
        **rows,
    }
    blob = json.dumps(payload, indent=2, ensure_ascii=False)

    archive = HERE / "results" / f"{stamp}.json"
    archive.parent.mkdir(exist_ok=True)
    archive.write_text(blob, encoding="utf-8")

    out = HERE / "results_latest.json"
    out.write_text(blob, encoding="utf-8")
    print(f"\nเก็บผลดิบไว้ที่ {out.relative_to(ROOT)}"
          f" และ {archive.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
