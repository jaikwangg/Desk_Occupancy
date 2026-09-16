#!/usr/bin/env python3
"""
สร้างชุดทดสอบ "คนนั่งทำงานอยู่หน้าจอ" จาก Open Images V6/V7

**ชุดนี้วัดได้อย่างเดียวคือ `recall`** — อ่านหัวข้อ "สิ่งที่ชุดนี้ทำไม่ได้" ก่อนใช้

ทำไมต้องมีทั้งที่มี `desk_testset.json` (COCO) อยู่แล้ว — เพราะชุด COCO ได้แค่
**16 ภาพ / คน 25** เล็กเกินกว่าจะสรุปอะไรได้แน่น (ความต่าง "92% vs 100%" คือคน
แค่ 2 คน) ชุดนี้ให้ **คน 531 คน** จึงทำให้ความต่างระดับ 2-3 จุดมีความหมายขึ้นมา

ลิขสิทธิ์ (ต่างจาก dataset อื่นที่สำรวจแล้วไม่ผ่าน — ดู context.md ข้อ 14):
  - annotation ของ Open Images = **CC BY 4.0** (ใช้เชิงพาณิชย์ได้ ต้องอ้างอิง)
  - ตัวภาพเป็นของเจ้าของแต่ละคนบน Flickr ส่วนใหญ่ **CC BY 2.0**
  - repo เก็บแต่ "รายการ + พิกัดกล่อง + URL" ไม่ได้ commit ตัวภาพ

===========================================================================
สิ่งที่ชุดนี้ทำไม่ได้ — **ตรวจด้วยตาแล้วทั้งสองข้อ ไม่ใช่การเดา**
===========================================================================

1. **ใช้วัด false positive ไม่ได้** — กล่องคนของ Open Images **ไม่ครบ**
   แม้ในภาพที่ image-level label ยืนยันว่า "มีคน" (397 จาก 400 ภาพในชุดนี้
   ยืนยันแล้ว) เปิดดู FP ของ yolo11n 66 จุด พบว่าเกือบทั้งหมดตกบน**คนจริงที่
   ไม่มีกล่อง GT** เช่นภาพที่มีคน 3 คนแต่ label ไว้คนเดียว
   ตัวเลข FP ที่ได้จึงวัด "ช่องโหว่ของ annotation" ไม่ใช่ความผิดของโมเดล

2. **ไม่มีภาพ negative (โต๊ะว่าง)** — เคยลองใช้ image-level label ที่ระบุว่า
   คลาส Person มี `Confidence=0` เป็น "ยืนยันว่าไม่มีคน" **ซึ่งผิด**
   เปิดดู 17 ภาพที่ `rtmdet-n-person` (โมเดล FP ต่ำสุด) เด้ง พบว่ามีคนจริง
   แทบทุกใบ — เด็กสองคนหน้าแล็ปท็อป มือพิมพ์คีย์บอร์ด คนยืนข้างเปียโน
   `Confidence=0` แปลว่า "ผู้ตรวจตอบว่าป้ายนี้ไม่ตรง" ซึ่งกับภาพที่เห็นแค่
   มือ/แขนคนมักถูกตอบว่าไม่ใช่คน **ใช้ชุด `desk` (COCO) สำหรับภาพโต๊ะว่างแทน**
   เพราะ COCO annotate ครบทุกคลาสทุกภาพ "ไม่มีกล่องคน" จึงแปลว่าไม่มีคนจริง

3. **แก้เรื่องมุมกล้องไม่ได้** — ยังเป็นภาพถ่ายที่คนถ่าย ไม่ใช่เฟรมจากกล้อง
   ที่ติดมองโต๊ะทำงาน และฉากเอียงไปทางงานประชุม/hackathon มากกว่าออฟฟิศ
   ตรวจสุ่ม 36 ใบพบปนเปื้อน ~14% (ฉากกลางแจ้ง, คนยืนพูดบนเวที)

===========================================================================

เกณฑ์คัด (ตรงกับ `build_desk_testset.py` เท่าที่ข้อมูลเอื้อ):

  A. ฉากต้องเป็นโต๊ะทำงานในอาคาร — ต้องมี Computer monitor/Laptop/Computer
     keyboard และต้องไม่มีหมวดกลางแจ้ง/กีฬา/ยานพาหนะ/เครื่องดนตรี/เตียง
     **หมายเหตุสำคัญ:** ใช้เกณฑ์แบบ "ถ้า*เจอ*คลาสที่ขัด ให้ตัด" เท่านั้น
     ห้ามใช้เกณฑ์แบบ "ต้อง*มี* label X" (เช่น "ต้องมีเก้าอี้") เพราะ
     Open Images ไม่ได้ annotate ครบทุกคลาส — ตอนลองใช้ตัดทิ้งไป 459 จาก 522 คน
  B. อุปกรณ์ต้องอยู่ "ตรงหน้า" คนนั้น (เกณฑ์เดียวกับชุด COCO)
  C. **ไม่มี keypoints ให้ใช้** ต่างจาก COCO จึงยืนยัน "ท่านั่ง" ตรงๆ ไม่ได้
     เหลือแค่สัดส่วนกล่อง (h/w <= 2.0 = ไม่ใช่คนยืน) — จุดที่อ่อนกว่าชุด COCO
  D. คนต้องสูง >= 12% ของภาพ (เท่าชุด COCO)
  E. ตัดคนที่ปรากฏบนหน้าจอ 2 ชั้น: แฟล็ก `IsDepiction` ของ Open Images
     (ครอบคลุมตุ๊กตา/รูปปั้น/โปสเตอร์ด้วย) **และ** เกณฑ์ "อยู่ในกล่องจอ >= 60%"
     แบบชุด COCO เพราะ `IsDepiction` จับไม่ครบ — สองชั้นรวมกันตัดออก 288 คน
     ตัด `IsGroupOf` ด้วย (กล่องครอบคนเป็นกลุ่ม ระบุตำแหน่งรายคนไม่ได้)

รัน:
    python benchmark/build_openimages_testset.py                # validation + test (~100 MB)
    python benchmark/build_openimages_testset.py --with-train   # + train (annotation 2.3 GB)
    python benchmark/build_openimages_testset.py --stats-only   # ดูสถิติ ไม่เขียนไฟล์
    python benchmark/build_openimages_testset.py --with-train --max-positives 1000
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "openimages_testset.json"
ANNDIR = HERE / "openimages_annotations"

V5 = "https://storage.googleapis.com/openimages/v5/"
V6 = "https://storage.googleapis.com/openimages/v6/"
IMG_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{iid}.jpg"

# ไฟล์ annotation ต่อ split — (ไฟล์กล่อง, ไฟล์ image-level label)
SPLIT_FILES = {
    "validation": ("validation-annotations-bbox.csv", V5),
    "test": ("test-annotations-bbox.csv", V5),
    "train": ("oidv6-train-annotations-bbox.csv", V6),
}

# ---- MID ของคลาสที่ใช้ (จาก class-descriptions-boxable.csv) ----
PERSON = "/m/01g317"
HUMAN = {PERSON, "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}  # Person Man Woman Boy Girl
WORK_DEVICES = {"/m/02522": "Computer monitor", "/m/01c648": "Laptop",
                "/m/01m2v": "Computer keyboard"}
DESK_CONTEXT = {"/m/01y9k5": "Desk", "/m/04bcr3": "Table", "/m/01mzpv": "Chair",
                "/m/0h8n5zk": "Kitchen & dining room table"}
TELEVISION = "/m/07c52"
SCREEN_LIKE = set(WORK_DEVICES) | {TELEVISION}

# A — เจอคลาสพวกนี้ = ไม่ใช่ฉากโต๊ะทำงานในอาคาร ตัดทั้งภาพ
# (เทียบเคียงรายการ OUTDOOR_SPORT ของ build_desk_testset.py)
OUTDOOR_SPORT = {
    "/m/0199g": "Bicycle", "/m/04_sv": "Motorcycle", "/m/0k4j": "Car",
    "/m/01bjv": "Bus", "/m/07jdr": "Train", "/m/07r04": "Truck",
    "/m/019jd": "Boat", "/m/0h9mv": "Tire",
    "/m/015qff": "Traffic light", "/m/01pns0": "Fire hydrant",
    "/m/02pv19": "Stop sign", "/m/015qbp": "Parking meter",
    "/m/0cvnqh": "Bench",
    "/m/018xm": "Ball", "/m/0jbk": "Animal", "/m/03k3r": "Horse",
    "/m/01xq0k1": "Cattle", "/m/07bgp": "Sheep", "/m/0bwd_0j": "Elephant",
    "/m/01dws": "Bear", "/m/0898b": "Zebra", "/m/03bk1": "Giraffe",
    "/m/0dv5r": "Camera", "/m/06_fw": "Skateboard", "/m/071qp": "Surfboard",
    "/m/06__v": "Baseball bat", "/m/018xm2": "Tennis racket",
    "/m/09tvcd": "Wine glass",
    # เครื่องดนตรี = ฉากเวที/ห้องซ้อม ไม่ใช่โต๊ะทำงาน (รอบแรกได้นักไวโอลินบนเวที)
    "/m/04szw": "Musical instrument", "/m/07y_7": "Violin", "/m/0342h": "Guitar",
    "/m/05r5c": "Piano", "/m/026t6": "Drum",
    # เตียง/โซฟา = นั่งเล่นในบ้าน ไม่ใช่โต๊ะทำงาน (รอบแรกได้คนนอนเล่นแล็ปท็อปบนเตียง)
    "/m/03ssj5": "Bed", "/m/03m3pdh": "Sofa bed", "/m/026qbn5": "Studio couch",
}

MIN_PERSON_H = 0.12
ON_SCREEN_FRAC = 0.60   # E — อยู่ในกล่องจอเกินเท่านี้ = คนที่ปรากฏบนหน้าจอ
OCCL_MIN = 0.08
MAX_ASPECT = 2.0
DEDUP_IOU = 0.60   # Person กับ Man/Woman มักครอบคนเดียวกัน — รวมเป็นกล่องเดียว


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  โหลด {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def ensure_split(split: str) -> Path:
    """โหลดเฉพาะไฟล์กล่อง — ไฟล์ image-level label ไม่ได้ใช้แล้ว
    (เคยใช้หาภาพ "ยืนยันว่าไม่มีคน" แต่พิสูจน์แล้วว่าเชื่อไม่ได้ ดู docstring)"""
    box_name, box_base = SPLIT_FILES[split]
    box = ANNDIR / box_name
    download(box_base + box_name, box)
    return box


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = min(ax + aw, bx + bw) - max(ax, bx)
    dy = min(ay + ah, by + bh) - max(ay, by)
    if dx <= 0 or dy <= 0:
        return 0.0
    inter = dx * dy
    return inter / (aw * ah + bw * bh - inter)


def inter_area(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = min(ax + aw, bx + bw) - max(ax, bx)
    dy = min(ay + ah, by + bh) - max(ay, by)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def in_front_of(person: list[float], dev: list[float]) -> bool:
    """B — เกณฑ์เดียวกับ build_desk_testset.in_front_of() ทุกบรรทัด"""
    px, py, pw, ph = person
    dx, dy, dw, dh = dev
    dcx, dcy = dx + dw / 2, dy + dh / 2
    return (inter_area(person, dev) > 0
            and px <= dcx <= px + pw
            and (py + 0.20 * ph) < dcy < (py + ph + 0.5 * ph))


def dedup(boxes: list[list[float]]) -> list[list[float]]:
    """Person/Man/Woman ครอบคนเดียวกันได้ — เก็บกล่องใหญ่สุดของแต่ละกลุ่ม"""
    kept: list[list[float]] = []
    for b in sorted(boxes, key=lambda x: -x[2] * x[3]):
        if not any(iou(b, k) >= DEDUP_IOU for k in kept):
            kept.append(b)
    return kept


KEEP_CLASSES = HUMAN | SCREEN_LIKE | set(DESK_CONTEXT) | set(OUTDOOR_SPORT)

# หนึ่งกล่อง = tuple 7 ช่อง ไม่ใช่ dict ของ csv.DictReader
# (train มีกล่องที่เข้าเกณฑ์หลักล้าน ถ้าเก็บเป็น dict 13 คีย์จะกินแรมจนถูก OOM kill)
#   (label, x, y, w, h, is_depiction, is_group)
I_LBL, I_X, I_Y, I_W, I_H, I_DEP, I_GRP = range(7)


def load_split(split: str) -> dict:
    """อ่านไฟล์กล่อง 2 รอบเพื่อไม่ให้กินแรม — train มี 14.6 ล้านแถว
    รอบแรกเก็บแค่ ImageID ที่มีอุปกรณ์ทำงาน รอบสองเก็บกล่องเฉพาะภาพเหล่านั้น
    และเก็บเป็น tuple ไม่ใช่ dict"""
    box_path = ensure_split(split)

    print(f"  อ่าน {box_path.name} รอบที่ 1 (หาภาพที่มีอุปกรณ์) ...", flush=True)
    wanted: set[str] = set()
    with open(box_path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        i_img, i_lbl = head.index("ImageID"), head.index("LabelName")
        for row in rd:
            if row[i_lbl] in WORK_DEVICES:
                wanted.add(row[i_img])
    print(f"    ภาพที่มีอุปกรณ์ {len(wanted)}", flush=True)

    print(f"  อ่าน {box_path.name} รอบที่ 2 ...", flush=True)
    rows: dict[str, list[tuple]] = defaultdict(list)
    with open(box_path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ci = {k: head.index(k) for k in
              ("ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax",
               "IsDepiction", "IsGroupOf")}
        for row in rd:
            lbl = row[ci["LabelName"]]
            if lbl not in KEEP_CLASSES or row[ci["ImageID"]] not in wanted:
                continue
            x1, x2 = float(row[ci["XMin"]]), float(row[ci["XMax"]])
            y1, y2 = float(row[ci["YMin"]]), float(row[ci["YMax"]])
            rows[row[ci["ImageID"]]].append(
                (lbl, x1, y1, x2 - x1, y2 - y1,
                 row[ci["IsDepiction"]] == "1", row[ci["IsGroupOf"]] == "1"))
    del wanted
    return rows


def build(split: str, reasons: defaultdict) -> list:
    rows = load_split(split)
    positives: list[dict] = []

    def box(a):
        return [a[I_X], a[I_Y], a[I_W], a[I_H]]

    for iid, anns in rows.items():
        present = {a[I_LBL] for a in anns}
        devs = [box(a) for a in anns if a[I_LBL] in WORK_DEVICES]
        if not devs:
            continue
        if present & set(OUTDOOR_SPORT):
            reasons["ฉากไม่ใช่โต๊ะทำงาน"] += 1
            continue
        # หมายเหตุ: ชุด COCO บังคับว่า "ต้องมีเก้าอี้หรือโต๊ะในภาพ" แต่ที่นี่ใช้ไม่ได้
        # เพราะ Open Images **ไม่ได้ annotate ครบทุกคลาสทุกภาพ** — ภาพโต๊ะทำงานจำนวนมาก
        # ไม่มีกล่อง Desk/Chair ทั้งที่เห็นโต๊ะชัดๆ เกณฑ์นี้ตัดทิ้ง 459 คนจาก 522
        # เกือบทั้งหมดเป็น false negative จึงใช้ OUTDOOR_SPORT (ตัดเมื่อ*เจอ*คลาสที่ขัด)
        # แทน ซึ่งปลอดภัยกว่าเพราะไม่ต้องพึ่งการ*มีอยู่*ของ label

        screens = [box(a) for a in anns if a[I_LBL] in SCREEN_LIKE]
        humans_all = [box(a) for a in anns if a[I_LBL] in HUMAN]

        # ---- ไม่มีภาพ negative จากชุดนี้ — ดู "สิ่งที่ชุดนี้ทำไม่ได้" ใน docstring ----
        if not humans_all:
            continue

        # ---- positive ----
        good = [box(a) for a in anns
                if a[I_LBL] in HUMAN and not a[I_DEP] and not a[I_GRP]]
        keep, occl = [], []
        for b in dedup(good):
            if b[3] < MIN_PERSON_H:                     # D
                reasons["เล็กเกินไป"] += 1
                continue
            parea = max(b[2] * b[3], 1e-9)
            # E — คนที่ปรากฏบนหน้าจอ (วิดีโอคอล/รูปบนมอนิเตอร์/ทีวี)
            # IsDepiction ของ Open Images จับได้บางส่วนแต่ไม่ครบ ต้องมีเกณฑ์นี้ซ้อน
            if any(inter_area(b, s) / parea >= ON_SCREEN_FRAC for s in screens):
                reasons["คนบนหน้าจอ"] += 1
                continue
            if b[2] > 0 and b[3] / b[2] > MAX_ASPECT:   # C
                reasons["สัดส่วนเหมือนคนยืน"] += 1
                continue
            if not any(in_front_of(b, d) for d in devs):  # B
                reasons["อุปกรณ์ไม่ได้อยู่ตรงหน้า"] += 1
                continue
            keep.append([round(v, 4) for v in b])
            if any(inter_area(b, s) / parea >= OCCL_MIN for s in screens):
                occl.append([round(v, 4) for v in b])

        if not keep:
            continue
        positives.append({
            "file": f"oi_{iid}.jpg",
            "url": IMG_URL.format(split=split, iid=iid),
            "kind": "positive", "norm": True, "split": split,
            "persons": keep,
            "occluded": occl,
            # รวมกล่องคนทุกแบบ (รวม depiction/group) ไว้หัก false positive
            # ไม่งั้นการที่โมเดลเจอ "คนบนจอ" จะถูกนับเป็น FP ทั้งที่ GT ก็เห็นว่ามีคนตรงนั้น
            "persons_all": [[round(v, 4) for v in b] for b in humans_all],
            "devices": [[round(v, 4) for v in d] for d in devs],
            "seated_via": ["aspect"] * len(keep),
        })
    return positives


MAX_POSITIVES = 400   # ดู docstring ของ cap_positives()


def cap_positives(pos: list[dict], limit: int) -> list[dict]:
    """จำกัดจำนวนภาพ positive — เลือกแบบ **กระจายสม่ำเสมอบน ImageID ที่เรียงแล้ว**

    ทำไมต้องจำกัด: `--with-train` ให้ภาพ positive ~1,850 ใบ / คน ~2,450 คน
    ซึ่งเกินความจำเป็นมาก — n ระดับ 500 คนก็พอให้ความต่าง 2-3 จุดมีความหมายแล้ว
    (ชุด COCO มี 25 คน จึงเป็นปัญหา) แต่ต้นทุนโหลดภาพกับเวลาวัดโตเป็นเส้นตรง

    ทำไมไม่สุ่มด้วย random: ต้องการให้**รันกี่ครั้งก็ได้ชุดเดิม** ไม่งั้นตัวเลข
    ข้ามรอบเทียบกันไม่ได้เลย ImageID ของ Open Images เป็น hash อยู่แล้ว
    การเรียงแล้วหยิบเว้นระยะจึงกระจายเหมือนสุ่มโดยไม่ต้องพึ่ง seed
    """
    if len(pos) <= limit:
        return pos
    pos = sorted(pos, key=lambda r: r["file"])
    step = len(pos) / limit
    return [pos[int(i * step)] for i in range(limit)]


def main(argv: list[str]) -> int:
    stats_only = "--stats-only" in argv
    limit = MAX_POSITIVES
    for i, a in enumerate(argv):
        if a == "--max-positives" and i + 1 < len(argv):
            limit = int(argv[i + 1])
    splits = ["validation", "test"]
    if "--with-train" in argv:
        splits.append("train")

    ANNDIR.mkdir(exist_ok=True)
    allpos: list[dict] = []
    reasons: defaultdict = defaultdict(int)
    for s in splits:
        print("")
        print(f"== split {s} ==")
        got = build(s, reasons)
        print(f"  positive {len(got)} ภาพ / คน "
              f"{sum(len(r['persons']) for r in got)}")
        allpos += got

    before = len(allpos)
    allpos = cap_positives(allpos, limit)
    if len(allpos) < before:
        print("")
        print(f"จำกัด positive จาก {before} เหลือ {len(allpos)} ภาพ "
              f"(เปลี่ยนด้วย --max-positives N)")

    np_ = sum(len(r["persons"]) for r in allpos)
    no = sum(len(r["occluded"]) for r in allpos)
    print(f"\nรวม: positive {len(allpos)} ภาพ / คนนั่งหน้าจอ {np_} คน "
          f"(ถูกอุปกรณ์บัง {no} คน)")
    print("     ไม่มีภาพ negative จากชุดนี้ — ใช้ชุด desk (COCO) สำหรับโต๊ะว่าง")
    print("\nคนที่ถูกคัดออกเพราะ:")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k:<26} {v}")

    if stats_only:
        return 0
    recs = allpos
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nเขียน {OUT.relative_to(HERE.parent)} แล้ว ({len(recs)} ภาพ)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
