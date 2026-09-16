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

### D-FINE ที่ export เป็น OpenVINO แล้ว

สร้างด้วย `python benchmark/export_dfine_openvino.py dfine-n` (ไม่เข้า git)

| โฟลเดอร์ | ขนาด | ms (× yolo11s) | recall ถูกบัง | FP |
|---|---|---|---|---|
| `dfine-n_openvino_model/` | 15.9 MB | 0.57x | **97%** | 4 |
| `dfine-n_int8_openvino_model/` | **5.4 MB** | 0.46x | **97%** | 6 |

เรียกใน benchmark ด้วยชื่อ `dfine-n-ov` / `dfine-n-ov-int8`

**สรุปสถานะ:** `dfine-n` + OpenVINO **เร็วกว่า `yolo11s` pytorch 1.76 เท่าและแม่นกว่า
3 จุด** แต่ยังไม่เปลี่ยน production เพราะ (ก) FP สูงกว่า (4 vs 2) ต้องจูน `conf` ก่อน
(ข) `desk_occupancy_multi.py` ยังโหลดโมเดลผ่าน `ultralytics` เท่านั้น
ส่วน `rtmdet-*` และ `crowdhuman-yolov8n` แพ้ทุกช่อง ตัดทิ้งได้

รายละเอียดอยู่ใน `context.md` ข้อ 10 และ `benchmark/example_run.md`

## pose_landmarker.task (fallback — MediaPipe Pose)

ใช้เฉพาะกรณีไม่มี `ultralytics`

    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
