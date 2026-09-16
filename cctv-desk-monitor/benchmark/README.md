# benchmark/

วัดโมเดลตรวจจับบุคคลกับฉากจริงที่ **คนถูกจอ/แล็ปท็อปบัง**

```bash
python benchmark/benchmark_models.py                      # ชุด default (7 โมเดล)
python benchmark/benchmark_models.py yolo11s.pt           # เจาะจงโมเดล
python benchmark/benchmark_models.py --all                # ทุกตัวที่รองรับ (13 โมเดล)
python benchmark/benchmark_models.py yolo11s.pt dfine-s rtmdet-tiny   # เทียบข้ามตระกูล
```

ครั้งแรกจะโหลดภาพ 46 ใบจาก COCO (~9 MB) มาไว้ใน `benchmark/images/` (ไม่เข้า git)
ส่วนน้ำหนักโมเดลที่ยังไม่มีจะโหลดลง `models/` ให้อัตโนมัติ (ก็ไม่เข้า git)

ผลดิบของรอบล่าสุดถูกเขียนทับไว้ที่ `benchmark/results_latest.json`
ตัวอย่างการรันจริงพร้อมผลลัพธ์เต็ม ดู **[example_run.md](example_run.md)**

## โมเดลที่รองรับ

เรียกด้วยชื่อสั้นได้ ระบบเลือก backend ให้เองตามชื่อ

| ชื่อที่ใช้เรียก | backend | หมายเหตุ |
|---|---|---|
| `yolo11n.pt` `yolo11s.pt` `yolov8n.pt` `yolov8s.pt` | ultralytics | ตระกูลหลักของโปรเจกต์ |
| `rtdetr-l.pt` | ultralytics | transformer ตัวใหญ่ |
| `crowdhuman-yolov8n` | ultralytics | yolov8n ที่ fine-tune บน **CrowdHuman** (เทรนเฉพาะคน + ฉากแออัด/ถูกบัง) |
| `models/yolo11s_openvino_model` | ultralytics | path ที่เป็น**โฟลเดอร์** = โมเดลที่ export เป็น OpenVINO |
| `dfine-n` `dfine-s` `dfine-m` | transformers | **D-FINE** — transformer detector รุ่นต่อจาก RT-DETR |
| `rtmdet-tiny` `rtmdet-s` | onnx + openvino | **RTMDet** (COCO) จาก mmdeploy |
| `rtmdet-n-person` `rtmdet-m-person` | onnx + openvino | RTMDet รุ่นที่เทรน**คลาสคนเดียว** |

`dfine-*` ต้องมี `transformers` (`pip install transformers`) ตัวอื่นใช้ของที่มีอยู่แล้ว
โดยเฉพาะ RTMDet ที่รันผ่าน **OpenVINO runtime ซึ่งอ่าน `.onnx` ได้ตรงๆ** จึงไม่ต้องลง
`onnxruntime` และไม่ต้องลง `mmcv`/`mmdet` (ซึ่งต้อง compile บน Windows)

## ชุดทดสอบมาจากไหน

`occlusion_testset.json` คัดจาก **COCO val2017** — ground truth เป็นของ COCO ไม่ใช่การนับเอง

เงื่อนไขการคัด: กล่อง `person` ต้องซ้อนทับกล่อง `tv`/`laptop`/`keyboard` **8–60%** ของพื้นที่ตัว

| เกณฑ์ | เหตุผล |
|---|---|
| สูง ≥ 12% ของภาพ | ตัดคนตัวจิ๋วไกลๆ ที่ไม่ใช่ "คนนั่งที่โต๊ะ" |
| ทับ ≥ 8% | ต้องถูกบังจริง ไม่ใช่แค่นั่งใกล้จอ |
| ทับ ≤ 60% | **สำคัญ** — ถ้าตัวคนอยู่ในกล่องจอเกือบทั้งตัว แปลว่าเป็น *คนที่ปรากฏบนหน้าจอทีวี* ซึ่ง COCO ก็ label เป็น person แต่ไม่ใช่คนจริงที่ถูกบัง เกณฑ์นี้ตัดออกไป 19 ภาพ |

ได้ **46 ภาพ / คน 131 คน (ถูกจอบัง 69 คน)**

## วิธีวัด

ตรงกับที่ระบบใช้จริง: ย่อภาพเป็น 480px แล้วเช็คว่า **center ของ detection ตกในกล่อง GT**
ของคนนั้นไหม (ระบบจริงก็แมป center เข้า ROI แบบเดียวกัน)

- `recall คนถูกบัง` = ตัวเลขที่สำคัญที่สุดสำหรับงานนี้
- `false pos` = detection ที่ไม่ตกในกล่องคนใดเลย (เทียบกับคนทุกขนาด)

ทุก backend ถูกวัดด้วยเงื่อนไขเดียวกัน — `conf=0.5`, **2 threads**, และจับเวลาครอบ
`preprocess + inference + postprocess` ทั้งก้อน (เฟรม BGR เข้า → list ของ center ออก)
ไม่ใช่เฉพาะ forward pass เพราะ preprocessing ของ transformer/ONNX ไม่ฟรี

## เพิ่มโมเดลใหม่เข้ามาวัด

backend คือคลาสที่ทำแค่สัญญาข้อเดียว — **รับเฟรม BGR คืน list ของ `(cx, cy)`
ในหน่วยพิกเซลของเฟรมนั้น** การแปลงกลับไปสเกลภาพเต็มกับการนับ recall/FP
`evaluate()` จัดการให้เอง

```python
class MyDetector:
    def __init__(self, weights):
        ...                     # โหลดโมเดล จำกัด thread ให้เท่ากับ THREADS

    def __call__(self, bgr):
        boxes = my_model(bgr)   # อะไรก็ได้
        return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
```

แล้วผูกชื่อเข้ากับ backend ใน `build_detector()` และเพิ่มชื่อลง `ALL_MODELS`

ถ้าโมเดลใหม่ต้อง letterbox เอง ดู `RTMDetOnnxDetector` เป็นตัวอย่าง — จุดที่พลาดง่ายคือ
**หารพิกัดที่ได้กลับด้วย scale ของ letterbox** ไม่งั้น center จะเพี้ยนไปจาก GT ทั้งหมด
วิธีเช็คเร็วๆ ว่า preprocessing ถูก: ถ้า `found` กับ `inside_gt` ใกล้เคียงกัน
แปลว่าพิกัดตรง ถ้า `found` เยอะแต่ `inside_gt` เกือบศูนย์ แปลว่าพิกัดเพี้ยน

## ข้อจำกัด

- COCO เป็นภาพระดับสายตา ไม่ใช่มุม CCTV มองลงมา — ลำดับอาจเปลี่ยนเมื่อติดกล้องจริง
  ข้อนี้กระทบโมเดลที่เทรนกับภาพกล้องวงจรปิด (`rtmdet-*-person`,
  OpenVINO person-detection) **มากเป็นพิเศษ** เพราะเป็นคนละโดเมนกับที่มันเทรนมา
- ภาพมีขนาดไม่เท่ากันทุกใบ ทำให้ ms สูงกว่าของจริง ~20-25%
- ms บนโน้ตบุ๊กแกว่งตามความร้อน/โหลด — **อัตราส่วนภายในการรันเดียวกันเชื่อถือได้กว่าค่าสัมบูรณ์**
