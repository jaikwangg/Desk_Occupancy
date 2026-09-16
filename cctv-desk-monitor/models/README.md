# models/

โมเดลไม่ได้ commit ลง git (ดู `.gitignore`) — ดาวน์โหลดเองก่อนรัน

## yolo11n.pt (หลัก — ตรวจจับบุคคล)

ultralytics ดาวน์โหลดให้อัตโนมัติครั้งแรกที่เรียก `YOLO("yolo11n.pt")`
หรือดาวน์โหลดมาวางเองที่ `models/yolo11n.pt`:

    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt

## yolo11s.pt (ตัวเลือกแม่นกว่า)

    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt

สลับได้ที่ `detector.yolo_model` ใน `occupancy.yaml` บรรทัดเดียว

| โมเดล | ขนาด | ms/เฟรม | recall คนถูกจอบัง | edge 4-core @2fps |
|---|---|---|---|---|
| `yolo11n.pt` (ค่าเริ่มต้น) | 5.6 MB | ~68 | 88% | ~12-15 กล้อง |
| `yolo11s.pt` | 19.3 MB | ~120 | 94% | ~6-8 กล้อง |

ตัวเลขมาจาก `benchmark/benchmark_models.py` วัดกับภาพจริง 46 ใบจาก COCO
ที่มีคนถูกจอ/แล็ปท็อปบังจริง — ดู `benchmark/README.md`

## โมเดลสำหรับ benchmark เท่านั้น (ไม่ได้ใช้ใน production)

`benchmark/benchmark_models.py` โหลดให้อัตโนมัติเมื่อเรียกด้วยชื่อสั้น ไม่ต้องโหลดเอง

| ชื่อที่ใช้เรียก | ไฟล์ | ที่มา |
|---|---|---|
| `crowdhuman-yolov8n` | `crowdhuman_yolov8n.pt` | [raghavendra24/crowdhuman-yolov8n](https://huggingface.co/raghavendra24/crowdhuman-yolov8n) |
| `rtmdet-tiny` `rtmdet-s` | `rtmdet-{tiny,s}-coco.onnx` | [bukuroo/RTMDet-ONNX](https://huggingface.co/bukuroo/RTMDet-ONNX) |
| `rtmdet-n-person` `rtmdet-m-person` | `rtmdet-{n,m}-person.onnx` | เหมือนกัน (เทรนคลาสคนเดียว) |
| `dfine-n` `dfine-s` `dfine-m` | cache ของ transformers | [ustc-community/dfine-*-coco](https://huggingface.co/ustc-community/dfine-small-coco) |

ผลการวัดดู `benchmark/example_run.md` — **สรุป: `yolo11s` ยังเป็นตัว deploy**
เพราะ `dfine-n` แม่นกว่า (ถูกบัง 97% vs 94%) แต่ยังช้ากว่า 2 เท่าเพราะยังไม่ได้
export OpenVINO ส่วน `rtmdet-*` และ `crowdhuman-yolov8n` แพ้ทุกช่อง
รายละเอียดเหตุผลอยู่ใน `context.md` ข้อ 10

## pose_landmarker.task (fallback — MediaPipe Pose)

ใช้เฉพาะกรณีไม่มี `ultralytics`

    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
