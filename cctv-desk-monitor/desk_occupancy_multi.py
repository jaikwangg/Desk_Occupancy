#!/usr/bin/env python3
"""
Desk Occupancy Monitor - multi-camera / multi-zone

ตรวจจับว่ามีคนอยู่ที่โต๊ะหรือไม่ (PRESENT / AWAY) จากกล้อง CCTV
ดูบริบทและคู่มือทั้งหมดใน context.md

กลยุทธ์หลัก (สำคัญ):
    ตรวจจับบุคคล 1 ครั้งบน "เต็มเฟรม" ที่ย่อขนาดแล้ว จากนั้นค่อยเช็คว่า
    center ของแต่ละคนตกอยู่ใน ROI ของโซนไหน
    *ไม่ใช่* crop ภาพแยกตรวจทีละโซน ซึ่งทำให้ของบนโต๊ะกลายเป็น false positive

การใช้งาน:
    python desk_occupancy_multi.py --config occupancy.yaml
    python desk_occupancy_multi.py --config occupancy.yaml --test-away 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template_string, request

PRESENT = "PRESENT"
AWAY = "AWAY"

# ต้องตั้งก่อน torch โหลด OpenMP: ดีฟอลต์ของ OpenMP คือ busy-wait (spin)
# thread ที่ทำงานเสร็จแล้วจะวนรอเปล่าๆ ทำให้ CPU พุ่งหลายร้อย % ทั้งที่
# inference ใช้จริงแค่ ~9% ของ 1 core ที่ 2 fps
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# ฟอนต์ Hershey ของ OpenCV วาดภาษาไทยไม่ได้ (ออกมาเป็น ????) ถ้ามี Pillow
# และฟอนต์ไทยในเครื่อง จะใช้วาด label บนวิดีโอแทน ไม่มีก็ถอยไปใช้ zone id
_THAI_FONTS = [
    r"C:\Windows\Fonts\leelawui.ttf",      # Leelawadee UI (Windows)
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
]

# console บน Windows ดีฟอลต์เป็น cp874/cp1252 ทำให้ log ภาษาไทยเพี้ยน
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# ไฟล์ที่หน้า /calibrate เขียน ROI ที่ลากเอาไว้ (โหลดทับ config อัตโนมัติตอนเริ่ม)
ZONES_FILE = Path("data/zones_calibrated.json")


# --------------------------------------------------------------------------
# ตัวตรวจจับบุคคล - YOLO11n เป็นหลัก, MediaPipe Pose / HOG เป็น fallback
# ทุกตัว detect() คืนค่าเป็น center แบบ normalized [0..1] เพื่อให้แมปกลับไป
# ยัง source pixels ได้โดยไม่ต้องสนใจว่าย่อขนาดไปเท่าไร
# --------------------------------------------------------------------------
class PersonDetector:
    name = "none"

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """คืน [(nx, ny, confidence), ...] - จุดกึ่งกลางของคนแต่ละคน"""
        raise NotImplementedError


class YoloDetector(PersonDetector):
    name = "yolo"

    def __init__(self, weights: str, conf: float, threads: int = 2) -> None:
        import torch
        from ultralytics import YOLO

        # YOLO11n ที่ 480px ใช้เวลาเท่ากันแทบทุกค่า thread (~44-60 ms/เฟรม)
        # การปล่อยให้ใช้ทุก core จึงไม่ได้เร็วขึ้น แต่กิน CPU เพิ่มหลายเท่า
        if threads > 0:
            torch.set_num_threads(threads)
        # โฟลเดอร์ = โมเดลที่ export เป็น OpenVINO (เร็วกว่า .pt ~1.7-2.6 เท่าบน CPU)
        # ต้องบอก task ให้ ultralytics เพราะโฟลเดอร์ไม่มีข้อมูลนี้ในตัว
        if Path(weights).is_dir():
            self.model = YOLO(weights, task="detect")
        else:
            self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        h, w = frame.shape[:2]
        out: list[tuple[float, float, float]] = []
        # classes=[0] คือคลาส "person" ของ COCO
        for res in self.model.predict(frame, classes=[0], conf=self.conf, verbose=False):
            for box in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append((((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, float(box.conf[0])))
        return out


class MediaPipeDetector(PersonDetector):
    name = "mediapipe-pose"

    def __init__(self, model_path: str, conf: float, max_poses: int = 5) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._vision = mp_vision
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=max_poses,
            min_pose_detection_confidence=conf,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        out: list[tuple[float, float, float]] = []
        for landmarks in result.pose_landmarks:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            out.append((float(np.mean(xs)), float(np.mean(ys)), 1.0))
        return out


class HogDetector(PersonDetector):
    """
    fallback สุดท้าย - ความแม่นยำต่ำกับคนนั่ง/มุมกล้องจากด้านบน

    หมายเหตุ: OpenCV 5.x ย้าย HOGDescriptor ออกจากโมดูลหลักแล้ว
    ถ้าใช้ cv2 5.x ตัวนี้จะใช้ไม่ได้ ต้องติดตั้ง ultralytics (แนะนำอยู่แล้ว)
    """

    name = "hog"

    @staticmethod
    def available() -> bool:
        return hasattr(cv2, "HOGDescriptor")

    def __init__(self) -> None:
        if not self.available():
            raise RuntimeError(
                f"cv2 {cv2.__version__} ไม่มี HOGDescriptor (ถูกตัดออกใน OpenCV 5.x)"
            )
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        h, w = frame.shape[:2]
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        out: list[tuple[float, float, float]] = []
        for (x, y, rw, rh), score in zip(rects, weights):
            out.append(((x + rw / 2) / w, (y + rh / 2) / h, float(score)))
        return out


def _load_font(size: int = 17) -> Any:
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _THAI_FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


_FONT = _load_font()


def build_detector(cfg: dict[str, Any]) -> PersonDetector:
    """เลือก detector ตามลำดับ YOLO -> MediaPipe -> HOG"""
    det_cfg = cfg.get("detector") or {}
    pose_cfg = cfg.get("pose") or {}
    conf = float(det_cfg.get("confidence", pose_cfg.get("presence_confidence", 0.5)))

    weights = det_cfg.get("yolo_model", "models/yolo11n.pt")
    # ถ้าไม่มีไฟล์ในเครื่อง ส่งชื่อเปล่าให้ ultralytics ดาวน์โหลดเอง
    if not Path(weights).exists():
        weights = "yolo11n.pt"
    threads = int(det_cfg.get("torch_threads", 2))
    try:
        det = YoloDetector(weights, conf, threads)
        print(f"[detector] ใช้ YOLO ({weights}) conf={conf} torch_threads={threads}")
        return det
    except Exception as exc:  # noqa: BLE001 - ตั้งใจ fallback ทุกกรณี
        print(f"[detector] YOLO ใช้ไม่ได้ ({exc}) -> ลอง MediaPipe Pose")

    pose_model = pose_cfg.get("model", "models/pose_landmarker.task")
    if Path(pose_model).exists():
        try:
            det = MediaPipeDetector(pose_model, conf)
            print(f"[detector] ใช้ MediaPipe Pose ({pose_model})")
            return det
        except Exception as exc:  # noqa: BLE001
            print(f"[detector] MediaPipe ใช้ไม่ได้ ({exc}) -> ใช้ HOG")
    else:
        print(f"[detector] ไม่พบ {pose_model} -> ใช้ HOG")

    if HogDetector.available():
        print("[detector] ใช้ HOG (ความแม่นยำต่ำ - แนะนำให้ติดตั้ง ultralytics)")
        return HogDetector()

    raise SystemExit(
        "ไม่มี detector ที่ใช้ได้เลย\n"
        "  - ติดตั้งตัวหลัก:  pip install ultralytics\n"
        f"  - HOG ใช้ไม่ได้เพราะ OpenCV {cv2.__version__} ตัด HOGDescriptor ออกแล้ว (5.x)\n"
        f"  - MediaPipe ต้องมีไฟล์โมเดลที่ {pose_model} (ดู models/README.md)"
    )


# --------------------------------------------------------------------------
# โซนและ state machine
# --------------------------------------------------------------------------
@dataclass
class Zone:
    id: str
    name: str
    roi: list[int]  # [x1, y1, x2, y2] ใน source pixels
    away_seconds: float
    state: str = AWAY
    last_present_ts: float = 0.0
    last_change_ts: float = field(default_factory=time.time)
    person_count: int = 0

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, has_person: bool, now: float) -> str | None:
        """
        เดิน state machine แล้วคืนสถานะใหม่ถ้ามีการ "เปลี่ยน" (ไม่งั้นคืน None)

        has_person ต้องเป็น bool จริงๆ เท่านั้น - เคยมีบั๊กที่ส่ง dict เข้ามา
        แล้ว truthiness ทำให้ทุกโซนกลายเป็น PRESENT ตลอด
        """
        if not isinstance(has_person, bool):
            raise TypeError(f"has_person ต้องเป็น bool ไม่ใช่ {type(has_person).__name__}")

        if has_person:
            self.last_present_ts = now
            if self.state != PRESENT:
                self.state = PRESENT
                self.last_change_ts = now
                return PRESENT
            return None

        # ไม่เจอคน - รอให้ครบ away_seconds ก่อนค่อยเปลี่ยนเป็น AWAY
        # (กันคนลุกไปหยิบของแป๊บเดียว หรือเฟรมที่ตรวจพลาด)
        if self.state == PRESENT and (now - self.last_present_ts) >= self.away_seconds:
            self.state = AWAY
            self.last_change_ts = now
            return AWAY
        return None


# --------------------------------------------------------------------------
# บันทึก event - CSV + SQLite
# --------------------------------------------------------------------------
class EventLog:
    def __init__(self, csv_path: str, db_path: str, keep_in_memory: int = 200) -> None:
        self.lock = threading.Lock()
        self.recent: list[dict[str, Any]] = []
        self.keep = keep_in_memory

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(["ts", "camera_id", "zone_id", "zone_name", "state"])

        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_file), check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS events (
                   id        INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts        TEXT NOT NULL,
                   camera_id TEXT NOT NULL,
                   zone_id   TEXT NOT NULL,
                   zone_name TEXT,
                   state     TEXT NOT NULL
               )"""
        )
        self.db.commit()

    def write(self, camera_id: str, zone: Zone) -> None:
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        row = {
            "ts": ts,
            "camera_id": camera_id,
            "zone_id": zone.id,
            "zone_name": zone.name,
            "state": zone.state,
        }
        with self.lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([ts, camera_id, zone.id, zone.name, zone.state])
            self.db.execute(
                "INSERT INTO events (ts, camera_id, zone_id, zone_name, state) VALUES (?,?,?,?,?)",
                (ts, camera_id, zone.id, zone.name, zone.state),
            )
            self.db.commit()
            self.recent.append(row)
            del self.recent[: -self.keep]
        print(f"[event] {ts}  {camera_id}/{zone.id} -> {zone.state}")

    def latest(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock:
            return list(reversed(self.recent[-limit:]))


# --------------------------------------------------------------------------
# worker ต่อ 1 กล้อง
# --------------------------------------------------------------------------
class CameraWorker(threading.Thread):
    def __init__(
        self,
        index: int,
        cfg: dict[str, Any],
        detector: PersonDetector,
        events: EventLog,
        downscale_width: int,
    ) -> None:
        super().__init__(daemon=True, name=f"cam-{index}")
        self.index = index
        self.id: str = str(cfg.get("id", f"cam-{index}"))
        self.name_: str = str(cfg.get("name", self.id))
        self.source = cfg.get("source")
        self.fps = float(cfg.get("fps", 2) or 2)
        self.detector = detector
        self.events = events
        self.downscale_width = downscale_width

        self.zones: list[Zone] = [
            Zone(
                id=str(z["id"]),
                name=str(z.get("name", z["id"])),
                roi=[int(v) for v in z["roi"]],
                away_seconds=float(z.get("away_seconds", 60)),
            )
            for z in cfg.get("zones", [])
        ]

        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None  # เฟรมดิบล่าสุด (สำหรับ /snapshot)
        self.annotated: np.ndarray | None = None  # เฟรมที่วาดกรอบแล้ว (สำหรับ /video)
        self.detections: list[tuple[int, int]] = []  # center ใน source pixels
        self.stop_flag = threading.Event()

    # -- helpers ---------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        src = self.source
        if isinstance(src, str):
            if src.isdigit():
                src = int(src)
            elif src.startswith("rtsp"):
                # RTSP over TCP เสถียรกว่า UDP สำหรับกล้องในวง LAN
                os.environ.setdefault(
                    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
                )
        cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - backend บางตัวไม่รองรับ
            pass
        return cap

    def set_zones(self, zones: list[dict[str, Any]]) -> None:
        """ใช้โซนใหม่จากหน้า /calibrate ทันที (คงสถานะเดิมไว้ถ้า id ตรงกัน)"""
        with self.lock:
            old = {z.id: z for z in self.zones}
            new: list[Zone] = []
            for z in zones:
                zone = Zone(
                    id=str(z["id"]),
                    name=str(z.get("name", z["id"])),
                    roi=[int(v) for v in z["roi"]],
                    away_seconds=float(z.get("away_seconds", 60)),
                )
                if zone.id in old:
                    prev = old[zone.id]
                    zone.state = prev.state
                    zone.last_present_ts = prev.last_present_ts
                    zone.last_change_ts = prev.last_change_ts
                new.append(zone)
            self.zones = new

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        with self.lock:
            return [
                {
                    "camera_id": self.id,
                    "camera_name": self.name_,
                    "camera_index": self.index,
                    "zone_id": z.id,
                    "zone_name": z.name,
                    "state": z.state,
                    "roi": z.roi,
                    "person_count": z.person_count,
                    "seconds_in_state": round(now - z.last_change_ts, 1),
                }
                for z in self.zones
            ]

    def snapshot_jpeg(self) -> bytes | None:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def annotated_jpeg(self) -> bytes | None:
        with self.lock:
            frame = None if self.annotated is None else self.annotated.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    # -- วาดกรอบ ----------------------------------------------------------
    def _label(self, zone: Zone) -> str:
        """ถ้าวาด unicode ไม่ได้ ให้ใช้ zone id (ASCII) แทนชื่อไทย ดีกว่าโชว์ ????"""
        name = zone.name if _FONT is not None or zone.name.isascii() else zone.id
        return f"{name}: {zone.state}"

    def _annotate(self, frame: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
        out = frame.copy()
        labels: list[tuple[int, int, str, tuple[int, int, int]]] = []

        for z in self.zones:
            x1, y1, x2, y2 = z.roi
            color = (0, 200, 0) if z.state == PRESENT else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = self._label(z)
            cv2.rectangle(out, (x1, y1), (x1 + 9 * len(label) + 12, y1 + 26), color, -1)
            labels.append((x1 + 5, y1 + 4, label, color))

        for cx, cy in centers:
            cv2.circle(out, (cx, cy), 7, (0, 255, 255), -1)
            cv2.circle(out, (cx, cy), 7, (0, 0, 0), 1)

        if _FONT is not None:
            # วาดข้อความทั้งหมดทีเดียวผ่าน Pillow (รองรับภาษาไทย)
            from PIL import Image, ImageDraw

            img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            for x, y, text, _color in labels:
                draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
            out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            for x, y, text, _color in labels:
                cv2.putText(
                    out, text, (x, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )
        return out

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        interval = 1.0 / max(self.fps, 0.1)
        cap = self._open()
        if not cap.isOpened():
            print(f"[{self.id}] เปิด source ไม่ได้: {self.source}")
            return
        print(f"[{self.id}] เริ่มทำงาน source={self.source} fps={self.fps} zones={len(self.zones)}")

        is_file = isinstance(self.source, str) and not str(self.source).startswith("rtsp")
        fail_count = 0

        while not self.stop_flag.is_set():
            started = time.time()
            ok, frame = cap.read()

            if not ok or frame is None:
                if is_file:
                    # ไฟล์วิดีโอจบแล้ว - วนกลับไปต้นเรื่องเพื่อทดสอบต่อเนื่อง
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                fail_count += 1
                print(f"[{self.id}] อ่านเฟรมไม่ได้ ({fail_count}) - เชื่อมต่อใหม่ใน 2 วิ")
                cap.release()
                time.sleep(2)
                cap = self._open()
                continue
            fail_count = 0

            h, w = frame.shape[:2]
            # ตรวจจับ 1 ครั้งบนเต็มเฟรมที่ย่อแล้ว (ไม่ crop รายโซน)
            if w > self.downscale_width:
                scale = self.downscale_width / w
                small = cv2.resize(frame, (self.downscale_width, int(h * scale)))
            else:
                small = frame

            try:
                people = self.detector.detect(small)
            except Exception as exc:  # noqa: BLE001 - ไม่ให้ทั้ง worker ตายเพราะเฟรมเดียว
                print(f"[{self.id}] detect ล้มเหลว: {exc}")
                people = []

            # normalized -> source pixels
            centers = [(int(nx * w), int(ny * h)) for nx, ny, _ in people]

            now = time.time()
            changed: list[Zone] = []
            with self.lock:
                for z in self.zones:
                    count = sum(1 for cx, cy in centers if z.contains(cx, cy))
                    z.person_count = count
                    if z.update(count > 0, now) is not None:
                        changed.append(z)
                self.frame = frame
                self.detections = centers
                self.annotated = self._annotate(frame, centers)

            for z in changed:
                self.events.write(self.id, z)

            elapsed = time.time() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

        cap.release()


# --------------------------------------------------------------------------
# Dashboard (Flask)
# --------------------------------------------------------------------------
DASHBOARD_HTML = """<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Desk Occupancy Monitor</title>
<style>
 :root{--bg:#0f1419;--card:#1a212b;--line:#2b3441;--fg:#e6edf3;--muted:#8b98a9;
       --ok:#2ea043;--away:#d1242f}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px}
 header{display:flex;justify-content:space-between;align-items:center;
        flex-wrap:wrap;gap:12px;margin-bottom:16px}
 h1{font-size:19px;margin:0}
 a.btn{background:#2f81f7;color:#fff;text-decoration:none;padding:8px 14px;
       border-radius:7px;font-size:13px}
 .stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
 .stat{background:var(--card);border:1px solid var(--line);border-radius:9px;
       padding:12px 18px;min-width:110px}
 .stat b{display:block;font-size:24px;line-height:1.2}
 .stat span{color:var(--muted);font-size:12px}
 .grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
       margin-bottom:22px}
 .zone{background:var(--card);border:1px solid var(--line);border-left-width:4px;
       border-radius:9px;padding:12px 14px}
 .zone.PRESENT{border-left-color:var(--ok)}
 .zone.AWAY{border-left-color:var(--away)}
 .zone h3{margin:0 0 2px;font-size:15px}
 .zone .cam{color:var(--muted);font-size:12px}
 .zone .st{font-weight:700;margin-top:8px}
 .zone.PRESENT .st{color:var(--ok)} .zone.AWAY .st{color:var(--away)}
 .zone .meta{color:var(--muted);font-size:12px}
 .cams{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(380px,1fr))}
 .cams figure{margin:0;background:var(--card);border:1px solid var(--line);
              border-radius:9px;overflow:hidden}
 .cams img{width:100%;display:block;background:#000}
 .cams figcaption{padding:8px 12px;font-size:13px;color:var(--muted)}
 table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
 th{color:var(--muted);font-weight:500}
 h2{font-size:15px;margin:22px 0 0}
</style></head><body>
<header>
  <h1>Desk Occupancy Monitor</h1>
  <a class="btn" href="/calibrate">ตั้งค่าโซนโต๊ะ</a>
</header>
<div class="stats" id="stats"></div>
<div class="grid" id="zones"></div>

<h2>วิดีโอสด</h2>
<div class="cams">
  {% for c in cameras %}
  <figure>
    <img src="/video/{{ c.index }}" alt="{{ c.name }}">
    <figcaption>{{ c.name }} ({{ c.id }})</figcaption>
  </figure>
  {% endfor %}
</div>

<h2>Event log ล่าสุด</h2>
<table><thead><tr><th>เวลา</th><th>กล้อง</th><th>โซน</th><th>สถานะ</th></tr></thead>
<tbody id="events"></tbody></table>

<script>
async function refresh(){
  try{
    const [sum, st, ev] = await Promise.all([
      fetch('/summary').then(r=>r.json()),
      fetch('/status').then(r=>r.json()),
      fetch('/events').then(r=>r.json())
    ]);
    document.getElementById('stats').innerHTML = `
      <div class="stat"><b>${sum.total_zones}</b><span>โซนทั้งหมด</span></div>
      <div class="stat"><b style="color:#2ea043">${sum.PRESENT}</b><span>มีคน</span></div>
      <div class="stat"><b style="color:#d1242f">${sum.AWAY}</b><span>ไม่มีคน</span></div>`;
    document.getElementById('zones').innerHTML = st.map(z=>`
      <div class="zone ${z.state}">
        <h3>${z.zone_name}</h3>
        <div class="cam">${z.camera_name}</div>
        <div class="st">${z.state}</div>
        <div class="meta">${z.seconds_in_state}s · ตรวจพบ ${z.person_count} คน</div>
      </div>`).join('');
    document.getElementById('events').innerHTML = ev.map(e=>`
      <tr><td>${e.ts.replace('T',' ').slice(0,19)}</td><td>${e.camera_id}</td>
          <td>${e.zone_name}</td><td>${e.state}</td></tr>`).join('')
      || '<tr><td colspan="4" style="color:#8b98a9">ยังไม่มี event</td></tr>';
  }catch(err){ console.error(err); }
}
refresh(); setInterval(refresh, 2000);
</script></body></html>"""


CALIBRATE_HTML = """<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ตั้งค่าโซนโต๊ะ</title>
<style>
 :root{--bg:#0f1419;--card:#1a212b;--line:#2b3441;--fg:#e6edf3;--muted:#8b98a9}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px}
 header{display:flex;justify-content:space-between;align-items:center;
        flex-wrap:wrap;gap:12px;margin-bottom:8px}
 h1{font-size:19px;margin:0}
 a,button{font-size:13px}
 a.back{color:#2f81f7}
 .hint{color:var(--muted);margin-bottom:16px}
 .cam{background:var(--card);border:1px solid var(--line);border-radius:9px;
      padding:14px;margin-bottom:16px}
 .cam h2{font-size:15px;margin:0 0 10px}
 .wrap{position:relative;display:inline-block;max-width:100%}
 .wrap img{max-width:100%;display:block;border-radius:6px}
 .wrap canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
 .zlist{margin-top:10px;font-size:13px}
 .zrow{display:flex;gap:8px;align-items:center;padding:4px 0}
 .zrow input{background:#0f1419;border:1px solid var(--line);color:var(--fg);
             border-radius:5px;padding:4px 8px;font-size:13px}
 .zrow code{color:var(--muted)}
 .zrow button{background:none;border:1px solid var(--line);color:#d1242f;
              border-radius:5px;cursor:pointer;padding:3px 8px}
 .bar{position:sticky;bottom:0;background:var(--bg);padding:12px 0;
      border-top:1px solid var(--line);display:flex;gap:10px;align-items:center}
 .save{background:#2ea043;color:#fff;border:0;border-radius:7px;
       padding:10px 20px;cursor:pointer;font-size:14px}
 .msg{color:var(--muted)}
</style></head><body>
<header><h1>ตั้งค่าโซนโต๊ะ</h1><a class="back" href="/">&larr; กลับ dashboard</a></header>
<p class="hint">ลากกรอบบนภาพให้ตรงบริเวณที่ <b>ตัวคน/เก้าอี้</b> ปรากฏจริง
อย่าลากกว้างเกินไป ไม่งั้นคนเดินผ่านจะทำให้เป็น PRESENT</p>

{% for c in cameras %}
<div class="cam" data-index="{{ c.index }}" data-id="{{ c.id }}">
  <h2>{{ c.name }} <span style="color:var(--muted);font-weight:400">({{ c.id }})</span></h2>
  <div class="wrap">
    <img src="/snapshot/{{ c.index }}?t={{ ts }}" alt="{{ c.name }}">
    <canvas></canvas>
  </div>
  <div class="zlist"></div>
</div>
{% endfor %}

<div class="bar">
  <button class="save" onclick="save()">บันทึก &amp; ใช้งาน</button>
  <span class="msg" id="msg"></span>
</div>

<script>
const initial = {{ zones_json|safe }};
const cams = [...document.querySelectorAll('.cam')].map(setup);

function setup(el){
  const idx = +el.dataset.index;
  const img = el.querySelector('img');
  const cv = el.querySelector('canvas');
  const list = el.querySelector('.zlist');
  const cam = {el, idx, id: el.dataset.id, img, cv, list,
               zones: (initial[idx]||[]).map(z=>({...z})), nat:[0,0]};
  let drag = null;

  img.onload = () => { cam.nat = [img.naturalWidth, img.naturalHeight];
                       cv.width = img.clientWidth; cv.height = img.clientHeight; draw(cam); };
  if (img.complete) img.onload();
  window.addEventListener('resize', () => {
    cv.width = img.clientWidth; cv.height = img.clientHeight; draw(cam); });

  const pos = e => { const r = cv.getBoundingClientRect();
                     return [e.clientX-r.left, e.clientY-r.top]; };
  cv.onmousedown = e => { const [x,y]=pos(e); drag={x0:x,y0:y,x1:x,y1:y}; };
  cv.onmousemove = e => { if(!drag) return; [drag.x1,drag.y1]=pos(e); draw(cam,drag); };
  cv.onmouseup = e => {
    if(!drag) return;
    [drag.x1,drag.y1]=pos(e);
    const sx = cam.nat[0]/cv.width, sy = cam.nat[1]/cv.height;
    const roi = [Math.round(Math.min(drag.x0,drag.x1)*sx),
                 Math.round(Math.min(drag.y0,drag.y1)*sy),
                 Math.round(Math.max(drag.x0,drag.x1)*sx),
                 Math.round(Math.max(drag.y0,drag.y1)*sy)];
    drag = null;
    if (roi[2]-roi[0] < 20 || roi[3]-roi[1] < 20) { draw(cam); return; }  // กันลากพลาด
    const n = cam.zones.length + 1;
    cam.zones.push({id:`${cam.id}-desk-${n}`, name:`Desk ${n}`, roi, away_seconds:60});
    draw(cam);
  };
  return cam;
}

function draw(cam, drag){
  const ctx = cam.cv.getContext('2d');
  ctx.clearRect(0,0,cam.cv.width,cam.cv.height);
  const sx = cam.cv.width/(cam.nat[0]||1), sy = cam.cv.height/(cam.nat[1]||1);
  ctx.lineWidth = 2; ctx.font = '13px system-ui';
  cam.zones.forEach(z=>{
    const [x1,y1,x2,y2] = z.roi.map((v,i)=> i%2 ? v*sy : v*sx);
    ctx.strokeStyle='#2ea043'; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle='rgba(46,160,67,.18)'; ctx.fillRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle='#2ea043'; ctx.fillText(z.name, x1+4, y1+15);
  });
  if(drag){ ctx.strokeStyle='#2f81f7'; ctx.setLineDash([5,4]);
            ctx.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0);
            ctx.setLineDash([]); }
  cam.list.innerHTML = '';
  cam.zones.forEach((z,i)=>{
    const row = document.createElement('div'); row.className='zrow';
    row.innerHTML = `<input value="${z.name}" size="12">
                     <code>[${z.roi.join(', ')}]</code>
                     <input type="number" value="${z.away_seconds}" size="4" style="width:70px">
                     <span style="color:var(--muted)">วิ</span>
                     <button>ลบ</button>`;
    const [nameIn, awayIn] = row.querySelectorAll('input');
    nameIn.oninput = () => { z.name = nameIn.value; };
    awayIn.oninput = () => { z.away_seconds = +awayIn.value || 60; };
    row.querySelector('button').onclick = () => { cam.zones.splice(i,1); draw(cam); };
    cam.list.appendChild(row);
  });
}

async function save(){
  const body = {cameras: cams.map(c=>({index:c.idx, id:c.id, zones:c.zones}))};
  const msg = document.getElementById('msg');
  msg.textContent = 'กำลังบันทึก...';
  try{
    const r = await fetch('/save_zones', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    msg.textContent = j.ok ? `บันทึกแล้ว ${j.zones} โซน - ใช้งานทันที`
                           : `ผิดพลาด: ${j.error}`;
  }catch(e){ msg.textContent = 'ผิดพลาด: ' + e; }
}
</script></body></html>"""


def build_app(workers: list[CameraWorker], events: EventLog) -> Flask:
    app = Flask(__name__)

    def cam_list() -> list[dict[str, Any]]:
        return [{"index": w.index, "id": w.id, "name": w.name_} for w in workers]

    @app.get("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML, cameras=cam_list())

    @app.get("/calibrate")
    def calibrate() -> str:
        zones = {
            w.index: [
                {"id": z.id, "name": z.name, "roi": z.roi, "away_seconds": z.away_seconds}
                for z in w.zones
            ]
            for w in workers
        }
        return render_template_string(
            CALIBRATE_HTML,
            cameras=cam_list(),
            zones_json=json.dumps(zones),
            ts=int(time.time()),
        )

    @app.get("/summary")
    def summary() -> Any:
        rows = [r for w in workers for r in w.status()]
        return jsonify(
            {
                "total_zones": len(rows),
                PRESENT: sum(1 for r in rows if r["state"] == PRESENT),
                AWAY: sum(1 for r in rows if r["state"] == AWAY),
            }
        )

    @app.get("/status")
    def status() -> Any:
        return jsonify([r for w in workers for r in w.status()])

    @app.get("/events")
    def event_list() -> Any:
        return jsonify(events.latest(int(request.args.get("limit", 50))))

    @app.get("/snapshot/<int:i>")
    def snapshot(i: int) -> Any:
        if not 0 <= i < len(workers):
            return jsonify({"error": "ไม่พบกล้อง"}), 404
        jpeg = workers[i].snapshot_jpeg()
        if jpeg is None:
            return jsonify({"error": "ยังไม่มีเฟรม"}), 503
        return Response(jpeg, mimetype="image/jpeg")

    @app.get("/video/<int:i>")
    def video(i: int) -> Any:
        if not 0 <= i < len(workers):
            return jsonify({"error": "ไม่พบกล้อง"}), 404
        worker = workers[i]

        def stream():
            while True:
                jpeg = worker.annotated_jpeg()
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(max(1.0 / max(worker.fps, 0.1), 0.2))

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/save_zones")
    def save_zones() -> Any:
        payload = request.get_json(silent=True) or {}
        cameras = payload.get("cameras")
        if not isinstance(cameras, list):
            return jsonify({"ok": False, "error": "รูปแบบข้อมูลไม่ถูกต้อง"}), 400

        by_index = {w.index: w for w in workers}
        by_id = {w.id: w for w in workers}
        persisted: dict[str, list[dict[str, Any]]] = {}
        total = 0

        for cam in cameras:
            worker = by_index.get(cam.get("index")) or by_id.get(cam.get("id"))
            if worker is None:
                continue
            zones = cam.get("zones") or []
            try:
                worker.set_zones(zones)
            except (KeyError, TypeError, ValueError) as exc:
                return jsonify({"ok": False, "error": f"โซนไม่ถูกต้อง: {exc}"}), 400
            persisted[worker.id] = [
                {"id": z.id, "name": z.name, "roi": z.roi, "away_seconds": z.away_seconds}
                for z in worker.zones
            ]
            total += len(worker.zones)

        ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
        ZONES_FILE.write_text(
            json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[calibrate] บันทึก {total} โซน -> {ZONES_FILE}")
        return jsonify({"ok": True, "zones": total})

    return app


# --------------------------------------------------------------------------
def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not cfg.get("cameras"):
        raise SystemExit(f"{path}: ไม่มี cameras ใน config")
    return cfg


def apply_saved_zones(cfg: dict[str, Any]) -> None:
    """ทับ ROI จาก config ด้วยค่าที่ลากไว้ในหน้า /calibrate (ถ้ามี)"""
    if not ZONES_FILE.exists():
        return
    try:
        saved = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[calibrate] อ่าน {ZONES_FILE} ไม่ได้ ({exc}) - ใช้ค่าจาก config แทน")
        return
    for cam in cfg["cameras"]:
        zones = saved.get(str(cam.get("id")))
        if zones:
            cam["zones"] = zones
    print(f"[calibrate] โหลดโซนที่ตั้งค่าไว้จาก {ZONES_FILE}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Desk Occupancy Monitor (multi-camera)")
    ap.add_argument("--config", default="occupancy.yaml", help="ไฟล์ config YAML")
    ap.add_argument(
        "--test-away", type=float, default=None,
        help="override away_seconds ทุกโซน (สำหรับทดสอบกับวิดีโอตัวอย่าง)",
    )
    ap.add_argument("--no-dashboard", action="store_true", help="ไม่เปิด Flask dashboard")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    apply_saved_zones(cfg)

    if args.test_away is not None:
        for cam in cfg["cameras"]:
            for z in cam.get("zones", []):
                z["away_seconds"] = args.test_away
        print(f"[config] โหมดทดสอบ: away_seconds = {args.test_away}s ทุกโซน")

    storage = cfg.get("storage") or {}
    events = EventLog(
        storage.get("log_csv", "data/occupancy.csv"),
        storage.get("db", "data/occupancy.sqlite"),
    )

    detector = build_detector(cfg)
    downscale = int((cfg.get("pose") or {}).get("downscale_width", 480))

    workers = [
        CameraWorker(i, cam, detector, events, downscale)
        for i, cam in enumerate(cfg["cameras"])
    ]
    for w in workers:
        w.start()

    dash = cfg.get("dashboard") or {}
    if args.no_dashboard or not dash.get("enabled", True):
        print("[dashboard] ปิดอยู่ - กด Ctrl+C เพื่อหยุด")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        host = dash.get("host", "0.0.0.0")
        port = int(dash.get("port", 8080))
        app = build_app(workers, events)
        print(f"[dashboard] http://localhost:{port}/  (calibrate: /calibrate)")
        try:
            app.run(host=host, port=port, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            pass

    for w in workers:
        w.stop_flag.set()
    print("\nหยุดทำงานแล้ว")
    return 0


if __name__ == "__main__":
    sys.exit(main())
