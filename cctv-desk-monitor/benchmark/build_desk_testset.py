#!/usr/bin/env python3
"""
สร้างชุดทดสอบ "คนนั่งทำงานอยู่หน้าจอ" จาก COCO val2017

ทำไมต้องมีชุดนี้ — ชุดเดิม (`occlusion_testset.json`) คัดแค่ "คนที่ถูกจอบัง"
ซึ่งได้ภาพถ่ายระดับสายตาแบบคนถ่ายรูปคนมาเยอะ ไม่ใช่ฉากโต๊ะทำงาน
ชุดนี้บังคับเงื่อนไขเพิ่มว่า **ต้องเป็นคนที่นั่งอยู่ที่โต๊ะซึ่งมีอุปกรณ์ทำงาน**

เกณฑ์คัด (ทุกข้อต้องผ่าน) — ดู `benchmark/README.md` สำหรับเหตุผลแต่ละข้อ

  A. ฉากต้องเป็นโต๊ะทำงานในอาคาร:
     - ต้องมี `laptop` หรือ `keyboard` (`tv` เดี่ยวๆ ไม่นับ = ห้องนั่งเล่น)
     - ต้องมี `chair` หรือ `dining table` (COCO ใช้ dining table กับโต๊ะทำงานด้วย)
     - ต้องไม่มีหมวดกลางแจ้ง/กีฬา/ถนน (`bench` `tennis racket` `car` ...)
       ข้อนี้มาจากการตรวจด้วยตา: รอบแรกได้ม้านั่งสวน สนามเทนนิส สนามบาส ห้องน้ำ
  B. อุปกรณ์ต้องอยู่ "ตรงหน้า" คนนั้น: กล่องซ้อนกล่องคน + จุดกลางอยู่ในช่วง x
     ของคน + อยู่ต่ำกว่าหัวไหล่ (รอบแรกใช้แค่ "อยู่ใกล้กัน" ซึ่งหลวมเกินไป)
  C. ต้องนั่ง — ผ่านทางใดทางหนึ่ง และกล่องต้องไม่สูงผอมเกิน h/w <= 2.0:
     C1 `keypoints` ยืนยัน: ต้นขาหุบ **และ** ต้นขาชี้ไปข้างหน้า (ระยะแนวนอน
        สะโพก-เข่า > ระยะแนวดิ่ง) ข้อหลังจำเป็นเพราะรอบแรกนักเทนนิสเสิร์ฟกับ
        กรรมการบาสเดินผ่านเกณฑ์ — ขายกทำให้ระยะดิ่งสั้นเหมือนท่านั่ง
     C2 `cutoff`: มีไหล่แต่ไม่มีหัวเข่า **และ** ตัวคนยืดลงไปถึงระดับอุปกรณ์
        = ท่อนล่างถูกโต๊ะบัง ซึ่งเป็นการจัดเฟรมแบบ "คนนั่งหลังโต๊ะ" ที่ต้องการ
  D. คนต้องใหญ่พอ: สูง >= 12% ของภาพ (เท่าชุดเดิม)
  E. กัน "คนที่ปรากฏบนหน้าจอ": ถ้ากล่องคนอยู่ในกล่อง tv >= 60% ให้ทิ้งคนนั้น
     (COCO label คนในทีวีเป็น person ด้วย) — เกณฑ์เดียวกับชุดเดิม

**ภาพ negative** — เพิ่มเข้ามาใหม่ ชุดเดิมไม่มีเลย:
  ภาพที่มี `chair` + (`laptop`|`keyboard`) แต่ **ไม่มี person เลย** = โต๊ะว่าง
  ทุก detection บนภาพพวกนี้คือ false positive ล้วน ซึ่งวัด failure mode ที่แย่
  ที่สุดของระบบได้ตรงๆ: **รายงาน PRESENT ตอนไม่มีคน**

รัน:
    python benchmark/build_desk_testset.py                    # ใช้ annotation ที่ cache ไว้
    python benchmark/build_desk_testset.py --stats-only       # ดูสถิติ ไม่เขียนไฟล์
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "desk_testset.json"
ANNDIR = HERE / "coco_annotations"
ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

MIN_PERSON_H = 0.12    # D — สูงอย่างน้อย 12% ของภาพ
ON_SCREEN_FRAC = 0.60  # E — อยู่ในกล่อง tv เกินเท่านี้ = คนบนหน้าจอ
OCCL_MIN = 0.08        # ถูกอุปกรณ์บังอย่างน้อยเท่านี้ = นับเป็น occluded
THIGH_RATIO = 0.55     # C1 — หัวเข่าต่ำกว่าสะโพกได้ไม่เกินเท่านี้ของความยาวลำตัว
MAX_ASPECT = 2.0       # C — คนยืนจะสูงผอม (h/w > 2.2) คนนั่งกล่องจะกะทัดรัดกว่า

WORK_DEVICES = ("laptop", "keyboard")          # A — สัญญาณ "โต๊ะทำงาน"
DESK_CONTEXT = ("chair", "dining table")       # A — ต้องมีที่นั่ง/โต๊ะ ไม่ใช่ถืออุปกรณ์เดินไปมา
SCREEN_LIKE = ("tv", "laptop", "keyboard")     # ใช้คิด occlusion

# A — ถ้าเจอหมวดพวกนี้ในภาพ แปลว่าไม่ใช่ฉากโต๊ะทำงานในอาคาร ตัดทั้งภาพ
# (มาจากการตรวจด้วยตารอบแรก: ม้านั่งสวน สนามเทนนิส สนามบาส ถนน ห้องน้ำ หลุดเข้ามาเยอะ)
OUTDOOR_SPORT = (
    "bench", "sports ball", "tennis racket", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "skis", "snowboard", "frisbee", "kite",
    "bicycle", "motorcycle", "car", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "toilet",
    "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
)

# index ของ keypoint ตามลำดับของ COCO
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14


def ensure_annotations() -> tuple[Path, Path]:
    inst = ANNDIR / "instances_val2017.json"
    kp = ANNDIR / "person_keypoints_val2017.json"
    if inst.exists() and kp.exists():
        return inst, kp
    ANNDIR.mkdir(exist_ok=True)
    z = ANNDIR / "annotations_trainval2017.zip"
    if not z.exists():
        print(f"โหลด COCO annotations (~253 MB) ครั้งเดียว -> {z} ...")
        urllib.request.urlretrieve(ZIP_URL, z)
    with zipfile.ZipFile(z) as f:
        for name in ("annotations/instances_val2017.json",
                     "annotations/person_keypoints_val2017.json"):
            target = ANNDIR / Path(name).name
            if not target.exists():
                target.write_bytes(f.read(name))
    return inst, kp


def inter_area(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = min(ax + aw, bx + bw) - max(ax, bx)
    dy = min(ay + ah, by + bh) - max(ay, by)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def in_front_of(person: list[float], dev: list[float]) -> bool:
    """B — อุปกรณ์ต้องอยู่ "ตรงหน้า" คนนั้นจริง ไม่ใช่แค่อยู่ในภาพเดียวกัน

    รอบแรกใช้แค่ "กล่องอยู่ใกล้กัน" ซึ่งหลวมเกินไป — แล็ปท็อปบนโต๊ะอีกตัวหนึ่ง
    หรือคีย์บอร์ดที่คนอื่นใช้ก็ผ่าน ตอนนี้บังคับ 3 อย่าง:
      1. กล่องอุปกรณ์ต้องซ้อนกล่องคนจริง (ไม่ใช่แค่เฉียดๆ)
      2. จุดกลางอุปกรณ์ต้องอยู่ในช่วง x ของคน = อยู่ตรงหน้าเขา
      3. อยู่ต่ำกว่าหัวไหล่ และไม่ต่ำกว่าตัวคนลงไปมาก
    """
    px, py, pw, ph = person
    dx, dy, dw, dh = dev
    dcx, dcy = dx + dw / 2, dy + dh / 2
    overlaps = inter_area(person, dev) > 0
    aligned = px <= dcx <= px + pw
    right_height = (py + 0.20 * ph) < dcy < (py + ph + 0.5 * ph)
    return overlaps and aligned and right_height


def seated_by_keypoints(kp: list[float]) -> bool:
    """C1 — ต้นขาหุบ = นั่ง (ท่ายืนหัวเข่าจะอยู่ต่ำกว่าสะโพกมาก)"""
    def pt(i):
        return kp[3 * i], kp[3 * i + 1], kp[3 * i + 2]

    for sh, hip, knee in ((L_SHOULDER, L_HIP, L_KNEE), (R_SHOULDER, R_HIP, R_KNEE)):
        _, sy, sv = pt(sh)
        hx, hy, hv = pt(hip)
        kx, ky, kv = pt(knee)
        if not (sv > 0 and hv > 0 and kv > 0 and hy > sy):
            continue
        torso = hy - sy
        if torso <= 0:
            continue
        # ต้นขาหุบ
        if (ky - hy) >= THIGH_RATIO * torso:
            continue
        # และต้นขาต้องชี้ไปข้างหน้า/เข้าหากล้อง ไม่ใช่ขาที่ยกขึ้นกลางอากาศ
        # (รอบแรกนักเทนนิสเสิร์ฟกับกรรมการบาสเดินผ่านเกณฑ์ เพราะขายกทำให้
        #  ระยะดิ่งสะโพก-เข่าสั้นเหมือนท่านั่ง) วัดมุมต้นขาจากแนวดิ่ง:
        # นั่ง = ต้นขาเกือบขนานพื้น -> ระยะแนวนอนต้องชนะระยะแนวดิ่ง
        if abs(kx - hx) < abs(ky - hy):
            continue
        return True
    return False


def lower_body_hidden(kp: list[float]) -> bool:
    """C2 — มีไหล่แต่ไม่มีหัวเข่า = ท่อนล่างถูกโต๊ะบัง

    เกณฑ์นี้อ่อนที่สุด ใช้ได้เฉพาะเมื่อผ่าน in_front_of() แล้ว และตัวคนต้อง
    ยืดลงไปถึงระดับโต๊ะ (เช็คในลูปหลัก) ไม่งั้นจะได้ครอปครึ่งตัวของคนยืนมาด้วย
    """
    def v(i):
        return kp[3 * i + 2]

    has_shoulder = v(L_SHOULDER) > 0 or v(R_SHOULDER) > 0
    has_knee = v(L_KNEE) > 0 or v(R_KNEE) > 0
    return has_shoulder and not has_knee


def main(argv: list[str]) -> int:
    stats_only = "--stats-only" in argv
    inst_path, kp_path = ensure_annotations()

    print("อ่าน annotations ...")
    inst = json.loads(inst_path.read_text(encoding="utf-8"))
    kpj = json.loads(kp_path.read_text(encoding="utf-8"))

    cat = {c["name"]: c["id"] for c in inst["categories"]}
    imgs = {i["id"]: i for i in inst["images"]}

    by_img: dict[int, list[dict]] = defaultdict(list)
    for a in inst["annotations"]:
        by_img[a["image_id"]].append(a)

    # keypoints ใช้ ann id เดียวกับ instances จึงจับคู่ตรงๆ ได้
    kp_by_ann = {a["id"]: a["keypoints"] for a in kpj["annotations"]}

    person_id = cat["person"]
    dev_ids = {cat[n] for n in WORK_DEVICES}
    screen_ids = {cat[n] for n in SCREEN_LIKE}
    tv_id, chair_id = cat["tv"], cat["chair"]
    ctx_ids = {cat[n] for n in DESK_CONTEXT}
    bad_ids = {cat[n] for n in OUTDOOR_SPORT}

    positives, negatives = [], []
    reasons = defaultdict(int)

    for img_id, anns in by_img.items():
        img = imgs[img_id]
        ih, iw = img["height"], img["width"]

        persons = [a for a in anns if a["category_id"] == person_id]
        devs = [a for a in anns if a["category_id"] in dev_ids]
        screens = [a for a in anns if a["category_id"] in screen_ids]
        tvs = [a for a in anns if a["category_id"] == tv_id]
        chairs = [a for a in anns if a["category_id"] == chair_id]

        # ---- negative: โต๊ะว่าง มีเก้าอี้ + อุปกรณ์ แต่ไม่มีคนเลย ----
        if not persons and devs and chairs and not (
                {a["category_id"] for a in anns} & bad_ids):
            negatives.append({
                "file": img["file_name"], "url": img["coco_url"], "kind": "negative",
                "persons": [], "occluded": [], "persons_all": [],
                "devices": [[round(v, 1) for v in a["bbox"]] for a in devs],
                "seated_via": [],
            })
            continue

        if not persons or not devs:
            continue
        present = {a["category_id"] for a in anns}
        if present & bad_ids:              # A — ฉากกลางแจ้ง/กีฬา/ถนน
            reasons["ฉากไม่ใช่โต๊ะทำงาน"] += 1
            continue
        if not (present & ctx_ids):        # A — ต้องมีเก้าอี้หรือโต๊ะ
            reasons["ไม่มีเก้าอี้/โต๊ะในภาพ"] += 1
            continue

        keep, occl, seated_via = [], [], []
        for a in persons:
            if a.get("iscrowd"):
                continue                       # กล่อง crowd ระบุตำแหน่งรายคนไม่ได้
            box = a["bbox"]
            if box[3] < MIN_PERSON_H * ih:     # D
                reasons["เล็กเกินไป"] += 1
                continue
            if box[2] > 0 and box[3] / box[2] > MAX_ASPECT:   # C — สูงผอม = ยืน
                reasons["สัดส่วนเหมือนคนยืน"] += 1
                continue
            parea = max(box[2] * box[3], 1e-6)
            # E — คนที่ปรากฏบนหน้าจอทีวี
            if any(inter_area(box, t["bbox"]) / parea >= ON_SCREEN_FRAC for t in tvs):
                reasons["คนบนหน้าจอ"] += 1
                continue
            # B
            front = [d for d in devs if in_front_of(box, d["bbox"])]
            if not front:
                reasons["อุปกรณ์ไม่ได้อยู่ตรงหน้า"] += 1
                continue
            # C
            kp = kp_by_ann.get(a["id"])
            if not kp:
                reasons["ไม่มี keypoints"] += 1
                continue
            if seated_by_keypoints(kp):
                via = "keypoints"
            elif lower_body_hidden(kp) and any(
                    box[1] + box[3] >= d["bbox"][1] for d in front):
                # ตัวคนต้องยืดลงไปถึงระดับอุปกรณ์ ไม่ใช่ครอปครึ่งตัวของคนยืน
                via = "cutoff"
            else:
                reasons["ไม่ใช่ท่านั่ง"] += 1
                continue

            keep.append([round(v, 1) for v in box])
            seated_via.append(via)
            if any(inter_area(box, s["bbox"]) / parea >= OCCL_MIN for s in screens):
                occl.append([round(v, 1) for v in box])

        if not keep:
            continue

        positives.append({
            "file": img["file_name"], "url": img["coco_url"], "kind": "positive",
            "persons": keep,
            "occluded": occl,
            # คนทุกคนทุกขนาด ใช้หัก false positive ไม่ให้การเจอคนที่เราไม่นับถูกทำโทษ
            "persons_all": [[round(v, 1) for v in a["bbox"]] for a in persons],
            "devices": [[round(v, 1) for v in a["bbox"]] for a in devs],
            "seated_via": seated_via,
        })

    recs = positives + negatives
    np_ = sum(len(r["persons"]) for r in positives)
    no = sum(len(r["occluded"]) for r in positives)
    via = defaultdict(int)
    for r in positives:
        for v in r["seated_via"]:
            via[v] += 1

    print(f"\nภาพ positive {len(positives)} ใบ | คนนั่งหน้าจอ {np_} คน "
          f"(ถูกอุปกรณ์บัง {no} คน)")
    print(f"  ยืนยันท่านั่งจาก keypoints {via['keypoints']} คน / "
          f"ท่อนล่างถูกบัง {via['cutoff']} คน")
    print(f"ภาพ negative {len(negatives)} ใบ (โต๊ะว่าง เก้าอี้+อุปกรณ์ ไม่มีคน)")
    print(f"รวม {len(recs)} ภาพ\n")
    print("คนที่ถูกคัดออกเพราะ:")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v}")

    if stats_only:
        return 0
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nเขียน {OUT.relative_to(HERE.parent)} แล้ว")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
