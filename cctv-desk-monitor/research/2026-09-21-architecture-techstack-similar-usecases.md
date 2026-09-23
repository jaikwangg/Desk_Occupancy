# Deep research: สถาปัตยกรรม / เทคสแตก / use case ใกล้เคียง

**วันที่ค้น:** 2026-09-21  
**วิธี:** deep-research workflow — แตก 6 มุมค้น → ดึง 27 แหล่ง → สกัด 135 claim → verify แบบ adversarial 3 เสียงต่อ claim (ต้องได้ 2/3 ปฏิเสธถึงจะตก)  
**ผล:** verify 25 claim → **ผ่าน 14 / ตก 11** (ดูข้อที่ตกท้ายไฟล์ — สำคัญพอๆ กับข้อที่ผ่าน)

> **ถ้าอ่านไม่หมด** ข้ามไปส่วนสุดท้าย «แล้วยังไงต่อ — อ่านเทียบกับโปรเจกต์นี้» ท้ายไฟล์

> ไฟล์นี้เป็น **วัตถุดิบ** ยังไม่ใช่การตัดสินใจ — ดูกติกาใน [README.md](README.md)  
> ข้อสรุปที่ตัดสินใจแล้วอยู่ใน `../context.md`

---

## คำถามตั้งต้น

```
หัวข้อวิจัย: สถาปัตยกรรม เทคสแตก และ use case ใกล้เคียง สำหรับระบบ "Desk Occupancy Monitoring ด้วย CCTV" เพื่อนำมา implement ต่อ

บริบทโปรเจกต์ปัจจุบัน (ใช้เป็นกรอบในการประเมินว่าอะไร "นำมาใช้ได้จริง"):
- โจทย์: ตรวจว่าโต๊ะทำงานแต่ละตัวมีคนนั่งอยู่หรือไม่ (สถานะ PRESENT / AWAY ต่อโซน) ในออฟฟิศที่ไทย ตำแหน่งโต๊ะ fix ไม่ย้าย
- จงใจไม่วัด "ประสิทธิภาพการทำงาน" — วัดแค่ occupancy เชิงวัตถุวิสัย เพราะ "นั่งนิ่ง ≠ ไม่ทำงาน"
- สถาปัตยกรรมตอนนี้: กล้อง RTSP → edge box (CPU Intel, Windows) → person detection บนเต็มเฟรม (downscale 480px) → แมป center ของคน เข้า ROI ของแต่ละโซน → state machine PRESENT/AWAY → เก็บเฉพาะ event log (CSV/SQLite) ไม่เก็บวิดีโอดิบ → Flask dashboard
- โมเดลที่ใช้/วัดแล้ว: YOLOv8n, yolo11n, D-FINE (nano/small/medium) export เป็น OpenVINO IR, RTMDet, CrowdHuman-trained models, OpenVINO Model Zoo person-detection
- ข้อจำกัดที่เจอ: ไม่มี public dataset ที่มีทั้งมุมกล้อง desk-level + bbox คุณภาพ eval + license เชิงพาณิชย์ (LOAF/CEPDOF/WEPDTOF/PIROPO/IndoorCrowd/SCB ใช้ไม่ได้หมด), ขาด "ภาพโต๊ะว่าง" จากกล้องจริงซึ่งเป็นเกณฑ์ตัดสินหลัก (false PRESENT)
- ข้อจำกัดด้าน privacy: PDPA ไทย, privacy-by-design, หลีกเลี่ยง Re-ID / appearance embedding เพราะระบุตัวบุคคลได้

สิ่งที่อยากได้จากการวิจัย:
1. Reference architecture ของระบบ workplace/desk occupancy analytics ที่มีคนทำจริง — ทั้ง commercial (เช่น VergeSense, XY Sense, Density, Butlr, Disruptive Technologies, FlexWhere ฯลฯ) และ open-source — ว่าวางโครงยังไง เก็บ metric อะไร ส่ง data ออกยังไง
2. ทางเลือกเซนเซอร์นอกจากกล้อง และ trade-off เทียบกับ CV: PIR, thermal/IR array (เช่น Butlr, Panasonic Grid-EYE), mmWave radar (TI IWR/Infineon), ToF/depth, ultrasonic, desk pressure/capacitive sensor, WiFi/BLE probe, badge/calendar/booking data, laptop/network telemetry — ตัวไหนแม่นกว่า/ถูกกว่า/privacy ดีกว่าสำหรับ "โต๊ะรายตัว"
3. Edge inference stack ที่ควรพิจารณา: OpenVINO vs ONNX Runtime vs TensorRT vs Hailo-8/8L vs Google Coral vs Rockchip NPU vs Jetson Orin Nano — cost per camera, throughput, ความยากในการ deploy บน Windows/Linux
4. Pattern "ROI crop classifier แทน whole-frame detector" — มีใครทำจริงและวัดผลไหม แม่นแค่ไหน เทียบ latency/accuracy กับ detector, วิธี bootstrap label ด้วย teacher model (OWLv2, Grounding DINO, SAM) และ active learning
5. เทคนิคลด false positive "โต๊ะว่างแต่เด้ง PRESENT" — background subtraction/static-object filtering, temporal smoothing, tracking (ByteTrack), chair-vs-person disambiguation, ensemble/cascade
6. Privacy-preserving computer vision ในที่ทำงาน: on-device only, ใช้ depth/thermal แทน RGB, face/body blurring, edge-only aggregation, ข้อกำหนดกฎหมาย GDPR/PDPA/works-council ที่เคยทำให้โครงการแบบนี้ล่ม และแนวปฏิบัติที่ผ่าน
7. Use case ใกล้เคียงที่ยก pattern มาใช้ได้: parking space occupancy detection, retail shelf/queue monitoring, classroom attendance, meeting room utilization, library seat availability, restaurant table occupancy — โดยเฉพาะงานที่เป็น "fixed camera + fixed zone + binary occupancy" เพราะเป็นรูปปัญหาเดียวกันเป๊ะ

เน้นสิ่งที่มีตัวเลขวัดได้ (accuracy, latency, cost per desk/camera) และสิ่งที่ implement ได้จริงบน CPU edge box ไม่เอาแค่ marketing claim
```

---

## สรุปผู้บริหาร

งานวิจัยที่ผ่านการตรวจสอบยืนยันสามแนวทางหลัก: (1) สถาปัตยกรรมที่ vendor เชิงพาณิชย์ (VergeSense) ใช้จริงคือ push webhook เมื่อสถานะเปลี่ยน + pull REST/Analytics API สำหรับข้อมูลย้อนหลัง และที่สำคัญคือ "ไม่" ยุบ occupancy เป็น binary เดียว แต่แยก person_count กับ signs_of_life (ของใช้บนโต๊ะ) เป็นคนละฟิลด์ พร้อม state available|occupied + flag passively_occupied ซึ่งตอบ ambiguity "โต๊ะมีเจ้าของแต่เดินออก" ได้ตรงกว่า PRESENT/AWAY เดี่ยวๆ; (2) use case ที่รูปปัญหาเหมือนกันเป๊ะและมีตัวเลขมากที่สุดคือ parking space occupancy ซึ่งใช้ pattern "per-ROI crop binary classifier" (mask คงที่ → crop ต่อ zone → CNN จิ๋ว) รันบน CPU ล้วนได้จริง — Pi 5 ที่ 0.01 วิ/zone ด้วยโมเดล 158K params (0.64 MB) และได้ 96.6–97.0% หลัง fine-tune ด้วย pseudo-label จากกล้องหน้างาน ถูกและเร็วกว่ารัน detector เต็มเฟรมระดับ order of magnitude; (3) การ bootstrap label ด้วย teacher model ควรใช้ confidence threshold ต่ำ (~0.1–0.2) เพราะ auto-label recall คือตัวทำนาย downstream accuracy ที่ดีที่สุด แต่คุณภาพ pseudo-label ตกหนักเมื่อมุมกล้องหลุด distribution ของ teacher (BDD gap -0.164 mAP50 เทียบ -0.035/-0.041 บน COCO/VOC) ซึ่งเป็นความเสี่ยงตรงของมุม CCTV เฉียง/สูงในออฟฟิศ. สำหรับการกด false PRESENT มี pattern production-grade จาก Frigate ให้ลอก: แยก min_score (กรองรายเฟรม) ออกจาก threshold (เทียบกับ median ของ score history) บวก geometry filter (area/ratio). ส่วนคำถามเรื่องเซนเซอร์ทางเลือกและ edge accelerator แทบไม่มี claim ใดรอด — เหลือเพียงหลักฐานว่า PIR อ่าน "ว่าง" ทั้งที่มีคนนั่งนิ่ง (false-off) ซึ่งเป็นทิศทาง error ที่ผิดสำหรับโจทย์นี้โดยตรง.

---

## ข้อค้นพบที่ผ่านการตรวจสอบ (14 ข้อ)

### 1. Reference architecture เชิงพาณิชย์ (VergeSense) แยก data egress เป็นสองทางชัดเจน: webhook push เมื่อสถานะเปลี่ยน + REST/Analytics API สำหรับ query ย้อนหลัง/aggregate — ตรงกับที่โปรเจกต์มีอยู่แล้ว (event log + Flask dashboard) และบอกว่าควรเติมอะไร

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0

Vendor docs: "Webhooks - push real-time data automatically to your platform; REST/Analytics API - pull specific or historical metadata" พร้อมตัวอย่าง report ราย building/floor/room ต่อเดือน. /reference/events ระบุ webhook 3 แบบ: space_report, space_availability (fires when availability has changed), motion_detected. Support docs ยืนยันมี Send Frequency ให้ยิงเฉพาะตอน person count / signs-of-life เปลี่ยน คือ send-on-change ไม่ใช่ polling. Implement ได้ทันที: state machine ยิง POST เฉพาะตอน transition PRESENT<->AWAY, dashboard อ่านจาก SQLite. ข้อควรระวัง: ใน VergeSense ตัว push มาจาก cloud ไม่ใช่ sensor — เราจะ push จาก edge box ตรงๆ ซึ่งเป็นคนละ layer (analogy ใช้ได้ แต่ topology ไม่เหมือน).

**Source:**
- <https://vergesense.readme.io/reference/reference-getting-started>
- <https://vergesense.readme.io/reference/events>
- <https://support.vergesense.com/hc/en-us/articles/13299436969485-Managing-Your-Webhooks>

### 2. อย่าบีบ occupancy เป็น binary เดียว — ระบบเชิงพาณิชย์แยก "คน" กับ "ของบนโต๊ะ" เป็นคนละฟิลด์ และมี state ว่าง/ไม่ว่าง แยกจาก flag passively_occupied

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0 (รวมสอง claim: [1] และ [2])

space_report payload: person_count (Number) = จำนวนคนในพื้นที่, signs_of_life (Boolean) = ตรวจพบ laptop/backpack/coat ฯลฯ, motion_detected (Boolean|null) — สามฟิลด์แยกกันใน payload เดียว. space_availability webhook: state = available|occupied, passively_occupied = true|false|null. สิ่งที่นำมาใช้ได้: log "unattended belongings" เป็นสัญญาณแยก แทนที่จะยุบรวมเป็น PRESENT. คำเตือนสำคัญต่อ benchmark protocol ของโปรเจกต์: VergeSense นิยาม passive occupancy ว่า laptop/เสื้อบนโต๊ะว่าง = occupied โดยตั้งใจ — ถ้าลอกกติกานั้นมา false PRESENT (metric ตัดสินหลัก) จะพุ่งทันที. รับโครงสร้างฟิลด์ อย่ารับ rule. ข้อจำกัด: signs_of_life เป็น optional/licensed feature ("May be null if signs of life is not enabled") และคำว่า orthogonal เป็นการตีความของผู้วิจัย docs ไม่ได้ระบุว่า combination ใดถูกกฎหมาย (cross-product ดิบมี 6 แบบ).

**Source:**
- <https://vergesense.readme.io/reference/events>
- <https://vergesense.readme.io/reference/spaces-1>
- <https://vergesense.com/blog/signs-of-life>

### 3. กล้อง/sensor หนึ่งตัวรายงานหลาย zone อิสระ = topology เดียวกับ 1 CCTV คลุมหลาย desk ROI และเป็นรูปแบบที่ vendor ทำจริงในระดับโต๊ะ

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0

docs: "The payload is an array -- one record per space that a sensor reports on." และ "Spaces are the smallest unit that information is collected for." Spacewell (third-party integrator ไม่ใช่ marketing ของ VergeSense) ยืนยันอิสระ: "A VergeSense sensor might be installed to track multiple locations (eg if the detection area covers multiple desks). VergeSense will partition the count areas from this 1 device in the platform, to make sure that data from the separate count areas is gathered individually." สรุป: โครงปัจจุบัน (1 กล้อง -> หลาย ROI -> หลาย state machine) ตรงกับ industry practice ไม่ต้องรื้อ.

**Source:**
- <https://vergesense.readme.io/reference/events>
- <https://vergesense.readme.io/reference/spaces-1>
- <https://spacewell.atlassian.net>

### 4. Pattern ที่มีหลักฐานแข็งที่สุดสำหรับ fixed-camera + fixed-zone + binary occupancy คือ per-ROI crop classifier (mask คงที่ -> crop ต่อ zone -> CNN binary) ไม่ใช่ detector เต็มเฟรม และยังเป็น pattern ที่ใช้อยู่จนถึงงานปี 2024

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 2-1 (mechanism ยืนยัน verbatim; ข้อค้านเป็นเรื่อง scalability ไม่ใช่ correctness)

Amato et al. (Expert Systems with Applications 2017, peer-reviewed): "Pictures captured by cameras are filtered by a mask that identifies the various parking spaces. The mask was built manually once and for all... Every patch is then classified using the trained CNN, to decide whether the corresponding parking space is empty or busy. Only the binary result is transmitted." ยังมีชีวิตปี 2024 (arXiv 2410.14705 ใช้ pattern เดิมบน PKLot/CNRPark-EXT). งานที่ค้าน (2208.08220) นิยาม pattern นี้เหมือนกันเป๊ะ ("a mask-based method") แต่ค้านเรื่อง scalability: ต้อง retrain ต่อมุมกล้อง ต้อง annotate ใหม่เมื่อย้ายกล้อง (ยกตัวอย่าง 900 annotations) และ detector-based ของเขาเองได้แค่ "competitive results compared to traditional classification solutions" ไม่ได้ชนะ accuracy. ข้อค้านทั้งหมดไม่ applicable กับโปรเจกต์นี้ เพราะโต๊ะ fix ไซต์เดียว ROI วาดมือไว้แล้ว — นี่คือเงื่อนไขที่ pattern นี้ได้เปรียบที่สุด. ข้อจำกัด: หลักฐานทั้งหมดเป็น parking ไม่มี desk-level.

**Source:**
- <https://openportal.isti.cnr.it/data/2017/366883/2017_366883.preprint.pdf>
- <https://arxiv.org/pdf/2410.14705>
- <https://arxiv.org/abs/2208.08220>

### 5. ROI crop classifier รันบน CPU edge ได้สบาย: Pi 5 = 0.01 วิ/zone (โหลด+crop+classify ครบ PyTorch ไม่ optimize) -> 100 zones ~1 วินาที; ย้อนไป Pi 2 (2015, ARM Cortex-A7 ล้วน ไม่มี GPU) = 50 zones ใน ~15 วินาที

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0 (ทั้ง [6] และ [8])

2410.14705 Sec.V-A: "we tested the system's deployment on a Raspberry Pi 5... using Python, OpenCV, and Pytorch without considering any optimization... On average, it took 0.01 seconds to process each spot as occupied/empty. If we consider that a camera can cover 100 parking spaces... it will take 1 second to refresh the statuses." โมเดลที่จับเวลาคือ Custom net 158,914 params / 0.64 MB / input 32x32. Amato 2017: "On the Raspberry Pi 2, the classification of 50 parking spaces and the transmission of the results to a web server takes about 15 seconds" (Cortex-A7, 1GB RAM, Caffe; GPU ใช้เฉพาะตอน train). นัยต่อ edge box Intel CPU: ถ้าเปลี่ยนจาก D-FINE/YOLO เต็มเฟรมมาเป็น crop classifier ต่อโต๊ะ งบ compute ลดระดับ order of magnitude และ scale ตามจำนวนโต๊ะแบบเชิงเส้นที่ถูกมาก. Caveats: 1 วิ/100 zones เป็นการคูณ (0.01x100) ไม่ใช่วัดจริง 100 zones; ไม่รวมเวลา RTSP pull/decode (แต่เป็นต้นทุน per-frame ไม่ scale ตาม zone); 0.3 วิ/zone ของ Pi 2 รวมเวลาส่งผลขึ้น web server จึงเป็น upper bound; ตัวเลข Pi 5 รายงานเฉพาะ Custom net ไม่ได้รายงานสำหรับ MobileNetV3-Small student.

**Source:**
- <https://arxiv.org/pdf/2410.14705>
- <https://ieeexplore.ieee.org/document/10903389>
- <https://openportal.isti.cnr.it/data/2017/366883/2017_366883.preprint.pdf>

### 6. โมเดล student จิ๋วที่ distill จาก teacher ensemble แล้ว fine-tune ด้วย pseudo-label จาก "กล้องตัวจริง" ทำได้ 96.6-97.0% — แต่ชัยชนะเหนือ teacher มาจาก domain adaptation ไม่ใช่ distillation และกลับด้านบนอีก test set หนึ่ง

**ความเชื่อมั่น:** 🟡 กลาง · **โหวต:** 2-1

Peer-reviewed IEEE ICMLA 2024 โดยทีมเดียวกับที่สร้าง PKLot. Table I: MobileNetV3-Large 4,204,594 params/17.00MB; MNv3-Small 1,519,906/6.20MB; Custom (3 conv + 2 pool + dense) 158,914/0.64MB. Table III (weighted avg, n=7 วันของ pseudo-label): Teacher 95.3%+/-0.4, Custom fine-tuned 96.6%+/-0.5, MNv3-Small fine-tuned 97.0%+/-0.3 แต่ถ้าไม่ fine-tune: Custom 81.1%, MNv3-Small 91.7% (ต่ำกว่า teacher ~14 pp). สิ่งที่ต้องรู้ก่อนลอก: (a) ผลเป็น sample-weighted average และกลับด้านบน CNRPark-EXT (Teacher 96.4% vs Custom 91.2% ห่าง 5.2 pp; MNv3-Small fine-tune ทำให้แย่ลง 95.6%->95.4%) เพราะได้ pseudo-label แค่ ~1,277 ตัวอย่างต่อมุมกล้องใน 7 วัน; (b) สูตรคือ teacher inference ~7 วันต่อมุมกล้อง ได้ student 1 ตัวต่อ 1 มุมกล้อง; (c) 4.2M params คือ 1 network ไม่ใช่ ensemble — teacher จริงมี 4-10 ตัว. บทเรียน: student จิ๋วใช้ได้ต่อเมื่อมีข้อมูลจากกล้องหน้างานพอ ซึ่งตรงกับคอขวดที่โปรเจกต์เจออยู่ (ขาดภาพโต๊ะว่างจากกล้องจริง) — ข้อมูลหน้างานคือคอขวด ไม่ใช่สถาปัตยกรรมโมเดล.

**Source:**
- <https://arxiv.org/pdf/2410.14705>
- <https://ieeexplore.ieee.org/document/10903389>

### 7. ต้นทุน edge node แบบ crop-classifier ต่ำมาก: USD 177 ต่อกล้องที่คลุม 16 zones (~USD 11/zone) บน Raspberry Pi 4 — แต่เชื่อได้แค่ระดับ order of magnitude

**ความเชื่อมั่น:** 🟡 กลาง · **โหวต:** 2-1

arXiv 2412.01983v2 (ธ.ค. 2024): "We estimated the solution cost to be 177 United States Dollars (USD), composing a Raspberry Pi 4 Model B (4GB RAM) a surveillance camera (Raspberry Pi Camera Module 3), a power supply, a MicroSD Card and a weatherproof case" + "A single photo was sufficient to cover the entire parking lot, which contained 16 spaces". Latency วัดจริงบนเครื่อง (สุ่ม 80 ภาพ): Pi 4 YOLOv8n 1+/-0.02 วิ, YOLOv9e 16+/-0.2 วิ; Pi 3 YOLOv8n 2 วิ, YOLOv9e 92+/-9.3 วิ. ผู้เขียนรับ latency สูงได้เพราะ "a wait time up to five minutes is acceptable, since the parking lot usually do not suffer major differences in terms of occupation in this time window". ปัญหาที่ต้องบอกต่อ: BOM รายการย่อยรวมได้ 55+25+10+15+15 = 120 USD ขัดกับ 177 ในเนื้อความ และ paper ไม่อธิบายส่วนต่าง 57 USD -> ตัวเลข USD 11/zone เป็นเลขคำนวณเอง ไม่ใช่เลขที่ paper อ้าง (ที่ BOM จริงจะเป็น ~7.50/zone). อีกทั้งเป็น parking (มุมไกล zone density ต่ำ) ส่วน desk ต้องการ zone density สูงกว่าที่มุมใกล้กว่ามาก; ราคาฮาร์ดแวร์ ธ.ค. 2024 ล้าสมัยแล้ว.

**Source:**
- <https://arxiv.org/html/2412.01983v2>

### 8. Pattern ลด false PRESENT ที่ production ใช้จริง: แยก threshold สองชั้น — min_score ทิ้ง detection รายเฟรมทันที ส่วน threshold เทียบกับ median ของ score history ของ tracked object

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0

Docs (ยืนยันทั้งหน้า rendered และ markdown ใน dev branch): "Any detection below min_score will be immediately thrown out and never tracked because it is considered a false positive" และ "threshold is based on the median of the history of scores (padded to 3 values) for a tracked object" + "Computed Score: the median of the most recent score history at that moment. This is the value compared against threshold." นำมาใช้กับ state machine ปัจจุบันได้ตรงๆ: แทนที่จะให้ detection เดี่ยว trigger PRESENT ให้เก็บ score history ต่อ ROI แล้วใช้ median (pad ขั้นต่ำ 3 ค่า) ตัดสิน transition. ข้อควรระวัง: stage 1 (min_score) เป็น per-frame ล้วน ไม่ใช่ temporal — มีแต่ stage 2 ที่เป็น temporal; Frigate ทำงานบน tracked object ทั้งเฟรมพร้อม zone-level override ไม่ใช่ per-ROI crop จึงเป็น pattern เชิงแนวคิด ไม่ใช่ drop-in.

**Source:**
- <https://docs.frigate.video/configuration/object_filters/>
- <https://raw.githubusercontent.com/blakeblackshear/frigate/dev/docs/docs/configuration/object_filters.md>

### 9. Geometry filter (min/max area, min/max ratio) เป็นกลไกลด false positive ที่ shipped จริงใน NVR open-source — area ระบุเป็น pixel หรือสัดส่วนของเฟรมได้ (resolution-independent), ratio = width/height

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0

Docs: "min_area and max_area filter on the area of an objects bounding box and can be used to reduce false positives that are outside the range of expected sizes... These values can either be in pixels or as a percentage of the frame (for example, 0.12 represents 12% of the frame)." และ "Conceptually, a ratio of 1 is a square, 0.5 is a 'tall skinny' box, and 2 is a 'wide flat' box. If min_ratio is 1.0, any object that is taller than it is wide will be ignored." สำหรับโปรเจกต์: กล้องนิ่ง + โต๊ะ fix => คนนั่งที่โต๊ะ A มีช่วงขนาด bbox และ aspect ratio ที่คาดเดาได้ ตั้ง gate ต่อ ROI ได้เลย ช่วยตัดเก้าอี้/เงา/วัตถุที่ misclassify. ข้อจำกัดสำคัญ: docs ไม่ให้ตัวเลขว่าลด FP ได้กี่ % — เป็นหลักฐานว่า practice นี้ใช้จริง ไม่ใช่หลักฐานว่าช่วยแค่ไหน. issue #5674 ชี้ว่า filter ประเมินตอน detection/threshold time ไม่ได้ re-filter ทุกเฟรม ดังนั้นถ้าจะใช้กด static false PRESENT ต้องบังคับ apply ทุกเฟรมเอง.

**Source:**
- <https://docs.frigate.video/configuration/object_filters/>
- <https://github.com/blakeblackshear/frigate/issues/5674>
- <https://docs.trafficmonitor.ai/configuration/frigate-config>

### 10. PIR ต่อโต๊ะไม่เหมาะกับโจทย์นี้ เพราะ failure mode หลักคือ false-off — อ่านว่าว่างทั้งที่มีคนนั่งนิ่ง ซึ่งเป็นทิศทาง error ที่ระบบ PRESENT/AWAY ต้องเลี่ยงที่สุด และนับจำนวนคนไม่ได้

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0

Applied Computing and Informatics 17(2) survey: "A PIR sensor can only detect the presence/absence of an occupant and cannot count the number of people" และ "The main drawback of such type sensor is that it is mostly prone due to 'False-off' errors, i.e. the lights are switched off even occupants are present"; ตารางสรุประบุข้อจำกัด PIR = False-off error, binary output, ใช้กับ DCV ไม่ได้. ยืนยันอิสระจาก Sensors 24(5):1533 (MDPI 2024): "Traditional PIR sensors frequently fail to recognize stationary individuals, causing errors in occupancy counts." กลไก: PIR รับรู้การเปลี่ยนแปลงของ IR ไม่ใช่การมีอยู่ (จึงมี dual-tech PIR+ultrasonic ขายเพื่อแก้ข้อนี้โดยเฉพาะ). Caveat: PIR ไม่ได้เอียงไป AWAY อย่างเดียว — มี false-ON จากลม HVAC แสงแดดตรง แหล่งความร้อนใกล้เคียง จึง noisy สองทาง. งานวิจัย PIR รุ่นปรับปรุง (MI-PIR, motion-induced/actuated) อ้าง 99% บนคนนิ่ง แต่เป็น prototype multi-sensor ไม่ใช่ PIR commodity.

**Source:**
- <https://www.emerald.com/insight/content/doi/10.1016/j.aci.2018.12.001/full/html>
- <https://www.mdpi.com/1424-8220/24/5/1533>
- <https://inside.lighting>

### 11. การ bootstrap label ด้วย teacher model ควรใช้ confidence threshold ต่ำ (~0.1-0.2) ไม่ใช่สูง — auto-label recall เป็นตัวทำนาย downstream accuracy ที่ดีที่สุด ขัดกับสัญชาตญาณ "ตั้ง threshold สูงเพื่อเลี่ยง label สกปรก"

**ความเชื่อมั่น:** 🟢 สูง · **โหวต:** 3-0

arXiv 2506.02359 (Griffin/Gangwar/Sela/Corso, Voxel51 + UMich, มิ.ย. 2025) Sec.3.5 verbatim: "alpha=0.2 results in the best AL recall for all rows, best mAP50 for 9 rows... alpha=0.8 results in the best AL precision and the worst mAP50 for all rows" และ "high auto-label recall is the best single predictor of downstream model performance followed by high auto-label F1 score." ฐานหลักฐานหนัก: 445 training runs, 169 unique label sets, teacher 3 ตัว (YOLO-World, YOLOE, Grounding DINO-T), downstream 6 โมเดล, 4 dataset. Sweep ละเอียดชี้ optimum ที่ alpha ใน [0.1,0.2] (ไม่ใช่ยิ่งต่ำยิ่งดีไม่มีพื้น). นัยตรงต่อโปรเจกต์: ถ้าใช้ OWLv2/Grounding DINO ช่วย label desk crops ให้ตั้ง threshold ต่ำแล้วปล่อยให้ student เรียน อย่าใช้ teacher แบบ high-precision. แต่ต้อง validate ด้วย false-PRESENT metric ของตัวเอง เพราะ paper optimize mAP50 ซึ่ง objective อาจสวนทางกับการกด false positive บนโต๊ะว่าง. Caveats: ขอบเขตคือ open-vocab auto-labeling ทั้ง dataset (zero human label) ไม่ครอบคลุม semi-supervised self-training คลาสสิกที่ใช้ threshold สูง (Unbiased Teacher 0.7, Soft Teacher 0.9); ผู้เขียนทั้งหมดสังกัด Voxel51 ซึ่งขาย tooling ประเภทนี้; เป็น preprint v1 ไม่มี limitations section.

**Source:**
- <https://arxiv.org/pdf/2506.02359>

### 12. คุณภาพ pseudo-label ตกฮวบเมื่อมุมกล้องหลุด distribution ของ teacher — gap teacher->student บน BDD (-0.164 mAP50) แย่กว่า VOC/COCO (-0.041/-0.035) ราว 4 เท่า เป็นสัญญาณเตือนตรงสำหรับมุม CCTV เฉียง/สูงในออฟฟิศ

**ความเชื่อมั่น:** 🟡 กลาง · **โหวต:** 2-1

Table 11 (YOLO11n student, YOLO-World @ conf 0.2): VOC human 0.756 vs auto 0.715; COCO 0.496 vs 0.460; BDD 0.434 vs 0.271. เหตุผลที่ paper ให้: "BDD consists entirely of autonomous driving viewpoints that are not representative of the original AL model training data" (teacher pretrain บน Objects365/GQA/Flickr30k). การพยายามหักล้างด้วยการ tune ต่อ dataset ล้มเหลว — config ดีที่สุดบน BDD คือ GDINO-0.2 ที่ 0.280 ยังห่าง human 0.434 อยู่ -0.154 (~35% relative) จึงไม่ใช่การ cherry-pick. ข้อแม้ที่ลดน้ำหนักสำหรับโปรเจกต์นี้: (a) paper ไม่เคยทดสอบมุม surveillance/overhead — การเหมาไปถึง office CCTV เป็น extrapolation คำของ paper เองอ่อนกว่า ("carefully consider the cost-performance trade-offs"); (b) viewpoint เป็นสมมติฐานที่ไม่ได้ ablate — BDD ต่างที่ taxonomy วัตถุเล็กไกล ฉากกลางคืนด้วย; (c) BDD คือ 10-class driving ส่วนโจทย์นี้ single-class person ซึ่งเป็นคลาสที่ VLM pretrain corpus มีเยอะที่สุด gap จริงน่าจะเล็กกว่า; (d) teacher ที่ทดสอบไม่รวม OWLv2 และ SAM ซึ่งเป็นสองตัวที่โปรเจกต์พิจารณาอยู่; (e) บน LVIS gap มาจาก vocabulary/long-tail (-32% relative) คนละแกนกับ viewpoint.

**Source:**
- <https://arxiv.org/pdf/2506.02359>

### 13. Autodistill ทำให้ label schema ถูกขับด้วย text prompt แทน annotation schema ที่สร้างมือ — CaptionOntology map caption -> class name และกำหนดทั้งเนื้อหา dataset และสิ่งที่ student ทำนาย

**ความเชื่อมั่น:** 🟡 กลาง · **โหวต:** 3-0 (mechanism สูง แต่ประสิทธิผลของ caption เชิงความสัมพันธ์ยังไม่มีหลักฐาน)

README/docs: "an Ontology defines how your Base Model is prompted, what your Dataset will describe, and what your Target Model will predict" และ "A simple Ontology is the CaptionOntology which prompts a Base Model with text captions and maps them to class names." ยืนยันระดับโค้ด: base_model = GroundedSAM(ontology=CaptionOntology({"shipping container": "container"})) — key = prompt ที่ส่งให้ base model, value = label ที่เขียนลง dataset, รับ string อะไรก็ได้. ข้อจำกัดสำคัญที่สุด: docs ของ autodistill เองบอกว่า base models "are not perfect yet" และอาจต้อง "creative prompting and few-shotting" — ไม่มีหลักฐานวัดผลว่า caption เชิงความสัมพันธ์อย่าง 'person seated at desk' vs 'empty chair' จะแยกได้จริงบน Grounding DINO/GroundedSAM ซึ่งจูนมาสำหรับ noun phrase. ต้องทดสอบบนเฟรมจริงก่อนไว้ใจเป็น teacher. อีกข้อ: repo last push 2025-05-14 (~16 เดือน) ไม่ archived แต่ไม่ active; docs footer ระบุ 2024.

**Source:**
- <https://github.com/autodistill/autodistill>
- <https://docs.autodistill.com>

### 14. Privacy-by-design ที่ทำจริงและตีพิมพ์แล้ว: ประมวลผลบน edge ทั้งหมด ส่งออกเฉพาะผลลัพธ์ binary ไม่ส่งภาพ — ตรงกับสถาปัตยกรรม "เก็บเฉพาะ event log ไม่เก็บวิดีโอดิบ" ที่โปรเจกต์ทำอยู่

**ความเชื่อมั่น:** 🟡 กลาง · **โหวต:** 3-0 (แต่เป็นหลักฐานเชิงสถาปัตยกรรม ไม่ใช่หลักฐานทางกฎหมาย)

Amato et al. (peer-reviewed, ESWA 2017) Conclusions: "the only information that is sent to a central server for visualization is the binary output of the classification"; Abstract: ระบบ "performs this task in real-time directly on smart cameras, without using a central server." ฮาร์ดแวร์ deploy จริง: Pi 2 B ในกล่องกล้อง outdoor บนหลังคาอาคารตรงข้ามลานจอด. ข้อแม้ด้านความแม่นยำของถ้อยคำ: "no central server" หมายถึงไม่มี server ทำ inference — server ยังมีอยู่ ทำ visualization และทำ multi-camera occlusion fusion ("the highest weighted confidence value is selected" ฝั่ง server). ช่องว่างที่งานวิจัยรอบนี้ไม่ตอบเลย: ไม่มี claim ใดรอดเกี่ยวกับ GDPR/PDPA/works-council, ไม่มีกรณีศึกษาโครงการที่ถูกระงับ, ไม่มีหลักฐานเรื่อง depth/thermal แทน RGB หรือ face blurring — คำถามข้อ 6 ของโจทย์วิจัยแทบไม่ได้คำตอบ.

**Source:**
- <https://openportal.isti.cnr.it/data/2017/366883/2017_366883.preprint.pdf>

---

## ข้อควรระวังของงานวิจัยชิ้นนี้

1) หลักฐานเชิงตัวเลขเกือบทั้งหมดมาจาก parking ไม่ใช่ desk — ไม่มี claim ใดที่รอดให้ตัวเลข accuracy/latency/cost ระดับ desk-level จริง. Parking มี zone density ต่ำกว่า มุมมองไกลกว่า และวัตถุ (รถ) นิ่งและแยกจากฉากหลังง่ายกว่าคนนั่งบนเก้าอี้มาก การโอนตัวเลขมาตรงๆ จะ optimistic เกินจริง.

2) สามคำถามในโจทย์วิจัยแทบไม่ได้คำตอบ: (ก) เปรียบเทียบ sensor modality เชิงตัวเลข — claim ที่ให้ช่วง accuracy ต่อ modality (กล้อง 80-96%, PIR 79-98%, ultrasonic ~90%, RFID 62-93%, CO2 ถึง 94%, WiFi/BLE 59-94%, fusion 76-100%) ถูก refute 0-3 จึงไม่มีฐานเปรียบเทียบ thermal/mmWave/ToF/desk pressure เหลืออยู่เลย นอกจากข้อจำกัดของ PIR; (ข) edge inference stack — claim เรื่อง OpenVINO speedup ถูก refute ทั้งตัวเลข 7x และ 3x (0-3 ทั้งคู่) และไม่มีหลักฐานใดๆ เลยเกี่ยวกับ Hailo-8/8L, Coral, Rockchip NPU, Jetson Orin Nano ทั้งด้าน throughput และ cost per camera; (ค) กฎหมาย PDPA/GDPR/works-council ไม่มีหลักฐานเหลือเลย.

3) Claim เรื่อง "full-frame detect -> map box center into ROI ชนะ crop/mask" ถูก refute 0-3 รวมถึงตัวเลข 99.68% balanced accuracy / 0.9975 F1 / FNR 0.07% ด้วย. แปลว่าสถาปัตยกรรมปัจจุบันของโปรเจกต์ (detect เต็มเฟรมแล้ว map center เข้า ROI) ไม่มีหลักฐานที่ผ่านการยืนยันรองรับว่าดีกว่า crop classifier และหลักฐานที่รอดทั้งหมดเอียงไปทาง crop classifier. อย่าตีความว่า crop classifier ชนะแน่นอน — ตีความว่ายังไม่มีใครเทียบสองแนวทางนี้บน metric false-PRESENT ที่โปรเจกต์ใช้.

4) Claim เรื่อง cross-site generalization collapse (LBP-SVM 52.88% ข้ามไซต์) ถูก refute 0-3 และ claim เรื่อง mAlexNet เสีย accuracy แค่ ~1% ถูก refute 1-2. จึงอย่าอ้างว่า "โมเดลจิ๋วเพียงพอเสมอ" หรือ "ต้อง train บนกล้องหน้างานเท่านั้น" เป็นข้อเท็จจริงที่ยืนยันแล้ว แม้หลักฐานจาก 2410.14705 (fine-tune ช่วยจาก 81.1% -> 96.6%) จะชี้ทิศทางเดียวกัน. เช่นเดียวกับ claim ว่า pseudo-label ที่ threshold 0.9 ผิดแค่ 0.27% ก็ถูก refute 0-3.

5) ปัญหาการอ้างอิง: claim VergeSense สามข้อระบุ URL ผิด (อ้าง /reference/reference-getting-started แต่เนื้อหาจริงอยู่ที่ /reference/events และ /reference/spaces-1) ต้องแก้ก่อนนำไปอ้างในเอกสาร.

6) ความไม่สอดคล้องภายในแหล่ง: BOM ของ arXiv 2412.01983 รวมได้ 120 USD แต่เนื้อความเขียน 177 USD โดยไม่อธิบายส่วนต่าง 57 USD — ตัวเลข USD 11/zone จึงเชื่อได้แค่ระดับ order of magnitude.

7) Time-sensitivity: ราคาและ latency ฮาร์ดแวร์ (ธ.ค. 2024, ~21 เดือนที่แล้ว) ล้าสมัยแล้ว Pi 5 และ AI accelerator เปลี่ยนภาพไปมาก; autodistill repo ไม่มี push มา ~16 เดือน; arXiv 2506.02359 เป็น preprint v1 ยังไม่ peer-review และผู้เขียนทั้งหมดมี COI (Voxel51 ขาย auto-labeling tooling); Amato 2017 เป็นฮาร์ดแวร์ปี 2015 + Caffe จึงใช้เป็น lower bound ได้เท่านั้น.

8) Verifier สองรายรายงานว่า WebSearch budget หมด (200/200) จึงไม่ได้รัน contradiction search อิสระสำหรับ claim [11] (autodistill), [12] (BDD gap) และ [13] (threshold 0.2) — ใช้การตรวจ primary source เต็มฉบับแทน ซึ่งพิสูจน์ว่า "อ้างตรงตามต้นฉบับ" แต่ไม่พิสูจน์ว่า "ไม่มีงานอื่นค้าน".

---

## คำถามที่ยังไม่มีคำตอบ

- Crop classifier ต่อ ROI กับ full-frame detector + center mapping ตัวไหนให้ false PRESENT บนโต๊ะว่างต่ำกว่า บนมุม CCTV ออฟฟิศจริง? ไม่มีงานใดเทียบสองแนวทางนี้ด้วย metric นี้เลย ต้องทำ ablation เองบน desk testset ที่มีอยู่ — นี่คือการตัดสินใจสถาปัตยกรรมที่แพงที่สุดที่ยังไม่มีคำตอบ
- OWLv2 / Grounding DINO / SAM แยก 'คนนั่งที่โต๊ะ' ออกจาก 'เก้าอี้ว่าง + เสื้อคลุมพาดพนัก' ได้จริงแค่ไหนบนมุมกล้องเฉียง/สูง? CaptionOntology รับ caption เชิงความสัมพันธ์ได้ทางกลไก แต่ไม่มีหลักฐานวัดผล และ claim เรื่อง viewpoint shift เตือนว่ามุมนอก distribution ทำ pseudo-label พังหนัก ต้องวัดเองก่อนลงทุนสร้าง distillation pipeline
- Edge accelerator ตัวไหนคุ้มที่สุดต่อกล้อง (Hailo-8L vs Coral vs iGPU/NPU ของ Intel vs CPU ล้วน) และยังจำเป็นไหมถ้าย้ายไป crop classifier? หลักฐานชี้ว่า 158K-param net รันบน Pi 5 CPU ได้ 100 zone/วินาที ซึ่งอาจแปลว่าไม่ต้องซื้อ accelerator เลย แต่ไม่มี claim ที่รอดให้ตัวเลข throughput/cost ของ accelerator ใดมาเทียบ
- ควรรับโมเดลข้อมูลแบบ VergeSense (person_count + signs_of_life + passively_occupied) มาเต็มรูป หรือรับแค่โครงสร้างฟิลด์? การมีฟิลด์ 'ของบนโต๊ะ' แยกมีประโยชน์ แต่ถ้าผู้ใช้/ผู้บริหารตีความ passively_occupied = PRESENT ระบบจะกลับไปมี false PRESENT สูงโดยปริยาย ต้องตัดสินว่า semantic ที่ส่งถึง dashboard คืออะไรและใครกำหนด
- PDPA ไทยและแนวปฏิบัติ works-council กำหนดอะไรกับ CV-based desk monitoring บ้าง? งานวิจัยรอบนี้ไม่ได้คำตอบแม้แต่ claim เดียว เหลือแค่หลักฐานเชิงสถาปัตยกรรมว่า 'ส่งออกเฉพาะ binary จาก edge' เป็น pattern ที่มีคนตีพิมพ์ ยังขาดทั้งฐานกฎหมายและกรณีศึกษาโครงการที่ถูกระงับ

---

## ข้อที่ถูกตีตก (11 ข้อ) — อ่านด้วย

ข้อเหล่านี้ฟังดูน่าเชื่อแต่**ไม่ผ่าน**การตรวจสอบ เก็บไว้กันค้นซ้ำ/กันหลงเชื่อรอบหน้า

- **The survey reports quantitative accuracy ranges per sensor modality for non-residential occupancy detection: video cameras 80–96%, PIR 79–98%, ultrasonic ~90%, RFID 62–93%, CO2 up to 94%, WiFi/WLAN/Bluetooth 59–94%, and multi-sensor fusion 76–100% — meaning camera-based CV is not automatically the most accurate modality, but is the most expensive and the worst for privacy.**  
  _โหวต 0-3_ — https://www.emerald.com/insight/content/doi/10.1016/j.aci.2018.12.001/full/html
- **Post-processing ROI selection — running the detector on the full unmasked frame and then testing each predicted bounding box's center (x,y) against a pixel-wise reference mask — outperformed pre-processing ROI masking (graying out non-ROI pixels), because masking the input removes context the detector needs and produces many false positives. This is precisely the 'full-frame detect → map box center into fixed zone' architecture, and the paper gives empirical support for preferring it over cropping/masking.**  
  _โหวต 0-3_ — https://arxiv.org/html/2412.01983v2
- **On a fixed camera covering 16 parking spaces, the post-processed pixel-wise ROI pipeline with YOLOv9e reached 99.68% balanced accuracy, 0.9975 F1 and 99.76% accuracy, with a false-negative rate of 0.07% — i.e. fixed-camera + fixed-zone + binary occupancy is solvable to near-ceiling accuracy with an off-the-shelf YOLO detector plus geometric zone mapping, no per-zone classifier required.**  
  _โหวต 0-3_ — https://arxiv.org/html/2412.01983v2
- **Smaller YOLO variants (the nano-class models analogous to YOLOv8n/yolo11n used in this project) degrade under lighting variation and generate a larger number of false positives, while the largest variants give stable predictions — so model-size choice, not just ROI logic, is a primary driver of the false-occupied error mode.**  
  _โหวต 0-3_ — https://arxiv.org/html/2412.01983v2
- **Site-specific labels can be bootstrapped automatically instead of hand-annotated: a teacher ensemble pseudo-labels 7 days of images from the new (unseen) parking lot at a confidence threshold of 0.9, and only 0.27% of the resulting pseudo-labels were wrong; students fine-tuned on them exceed 96% accuracy.**  
  _โหวต 0-3_ — https://arxiv.org/pdf/2410.14705
- **A drastically shrunk CNN (mAlexNet: 3 conv + 2 FC layers, roughly 1/1340 the parameters of AlexNet's 60 million, i.e. ~45k params) loses at most ~1% accuracy versus AlexNet when train and test come from the same dataset — so model capacity can be traded away almost for free on a binary occupancy task.**  
  _โหวต 1-2_ — https://openportal.isti.cnr.it/data/2017/366883/2017_366883.preprint.pdf
- **Cross-site / cross-viewpoint generalization is the dominant failure mode, not in-domain accuracy: classic hand-crafted-feature baselines collapse to near chance when moved to a new site (LBP-SVM trained on CNRPark scores 52.88% accuracy / 0.391 AUC on PKLot vs mAlexNet's 90.38% / 0.989), and even AlexNet trained on PKLot2Days drops to 65.31% accuracy / 0.580 AUC on CNRPark-EXT — evidence that a model must be trained on patches from the actual installed cameras.**  
  _โหวต 0-3_ — https://openportal.isti.cnr.it/data/2017/366883/2017_366883.preprint.pdf
- **Ultralytics' own OpenVINO benchmarks show the nano YOLO model on an Intel Core Ultra X7 358H drops from 29.28 ms/image (PyTorch FP32 CPU) to ~4.09 ms/image with OpenVINO, i.e. roughly a 7x per-frame speedup from export alone — directly relevant to whether a CPU-only edge box can run person detection on multiple RTSP streams.**  
  _โหวต 0-3_ — https://docs.ultralytics.com/integrations/openvino
- **The documented headline claim for CPU-only deployment is more conservative than the per-device table: OpenVINO is stated to deliver up to 3x CPU speedup over PyTorch on Intel processors, so a 3x (not 7x) figure is the safer planning number for a plain Intel CPU edge box without iGPU/NPU offload.**  
  _โหวต 0-3_ — https://docs.ultralytics.com/integrations/openvino
- **Autodistill implements exactly the teacher-to-student bootstrap pattern the project needs for desk-occupancy labeling: a zero-shot foundation ('Base') model auto-labels unlabeled images, that generated dataset trains a small supervised ('Target') model, and the resulting distilled model runs at the edge — with no manual annotation step in the loop.**  
  _โหวต 0-3_ — https://github.com/autodistill/autodistill
- **Auto-labeling with a vision-language teacher model costs ~1/5,000 the time and ~1/100,000 the money of human annotation: across VOC+COCO+LVIS+BDD the paper reports 6,703 human labeling hours / $124,092.54 vs 1.27 GPU-hours / $1.18 for auto-labels. This makes teacher-model bootstrapping of a desk-occupancy training set economically trivial compared with manual bbox labeling.**  
  _โหวต 0-3_ — https://arxiv.org/pdf/2506.02359

---

## แหล่งข้อมูลทั้งหมด (27)

| แหล่ง | คุณภาพ | มุมที่ใช้ | จำนวน claim |
|---|---|---|---|
| <https://vergesense.readme.io/reference/reference-getting-started> | primary | Industry reference architectures & sensor trade-offs | 5 |
| <https://xysense.com/xysense-vs-other-sensors/> | blog | Industry reference architectures & sensor trade-offs | 5 |
| <https://www.occuspace.com/blog/6-workplace-occupancy-sensors-and-the-best-use-for-each> | blog | Industry reference architectures & sensor trade-offs | 5 |
| <https://www.emerald.com/insight/content/doi/10.1016/j.aci.2018.12.001/full/html> | primary | Industry reference architectures & sensor trade-offs | 5 |
| <https://arxiv.org/html/2412.01983v2> | primary | Academic / analogous fixed-zone binary occupancy | 5 |
| <https://arxiv.org/pdf/2410.14705> | primary | Academic / analogous fixed-zone binary occupancy | 5 |
| <https://openportal.isti.cnr.it/data/2017/366883/2017_366883.preprint.pdf> | primary | Academic / analogous fixed-zone binary occupancy | 5 |
| <https://docs.ultralytics.com/integrations/openvino> | primary | Edge inference benchmarks & cost per camera | 5 |
| <https://terminalbytes.com/best-hardware-for-frigate-nvr-2026/> | blog | Edge inference benchmarks & cost per camera | 5 |
| <https://github.com/blakeblackshear/frigate/discussions/7601> | forum | Edge inference benchmarks & cost per camera | 5 |
| <https://docs.frigate.video/configuration/object_filters/> | primary | Practitioner false-positive reduction & auto-labeling | 5 |
| <https://github.com/blakeblackshear/frigate/discussions/21756> | forum | Practitioner false-positive reduction & auto-labeling | 5 |
| <https://y-t-g.github.io/tutorials/bg-images-for-yolo/> | blog | Practitioner false-positive reduction & auto-labeling | 5 |
| <https://github.com/autodistill/autodistill> | primary | Practitioner false-positive reduction & auto-labeling | 5 |
| <https://arxiv.org/pdf/2506.02359> | primary | Practitioner false-positive reduction & auto-labeling | 5 |
| <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12656557/> | primary | Practitioner false-positive reduction & auto-labeling | 5 |
| <https://fortune.com/2017/08/18/barclays-desk-tracking-devices/> | secondary | Privacy, law and deployment failures | 5 |
| <https://www.employee-monitoring.net/compliance/employee-monitoring-laws-thailand> | blog | Privacy, law and deployment failures | 5 |
| <https://securiti.ai/blog/employee-data-thailand/> | blog | Privacy, law and deployment failures | 5 |
| <https://pointgrab.com/occupancy-sensors-and-privacy-a-gdpr-ready-guide-for-european-workplaces/> | blog | Privacy, law and deployment failures | 5 |
| <https://www.cpomagazine.com/data-privacy/uk-ico-opens-probe-into-barclays-for-employee-surveillance/> | secondary | Privacy, law and deployment failures | 5 |
| <https://arxiv.org/pdf/2007.04678> | primary | Privacy, law and deployment failures | 5 |
| <https://github.com/RexxarCHL/library-seat-detection> | primary | Open-source implementations & adjacent verticals | 5 |
| <https://github.com/Haseeeb21/restaurant-cv-insights> | blog | Open-source implementations & adjacent verticals | 5 |
| <https://supervision.roboflow.com/detection/tools/polygon_zone/> | primary | Open-source implementations & adjacent verticals | 5 |
| <https://docs.frigate.video/configuration/object_detectors/> | primary | Open-source implementations & adjacent verticals | 5 |
| <https://www.mdpi.com/1424-8220/23/17/7642> | primary | Open-source implementations & adjacent verticals | 5 |


---

## แล้วยังไงต่อ — อ่านเทียบกับโปรเจกต์นี้

_ส่วนนี้เป็นการตีความของเราเอง ไม่ใช่ข้อสรุปของงานวิจัย — แยกออกจากข้างบนโดยตั้งใจ_

### เอามาใช้ได้เลย (ไม่ต้องรอกล้องจริง)

1. **Two-stage threshold แบบ Frigate** (ข้อ 8) — แยก `min_score` (ทิ้งรายเฟรม) ออกจาก
   `threshold` ที่เทียบกับ **median ของ score history ต่อ ROI** (pad ขั้นต่ำ 3 ค่า)
   ตอนนี้ state machine ให้ detection เฟรมเดียว trigger `PRESENT` ได้ → เปลี่ยนเป็น median
   คือการกด false PRESENT ที่ถูกที่สุดที่ทำได้ทันที **และวัดผลได้ด้วย desk testset ที่มีอยู่แล้ว**
2. **Geometry gate ต่อ ROI** (ข้อ 9) — กล้องนิ่ง + โต๊ะ fix ⇒ คนนั่งที่โต๊ะแต่ละตัวมีช่วง
   area/aspect ratio ที่คาดเดาได้ ตั้ง min/max ต่อโซนได้เลย ช่วยตัดเก้าอี้/เงา/ของบนโต๊ะ
   ⚠️ Frigate ประเมิน filter ตอน detection time ไม่ได้ re-filter ทุกเฟรม — ของเราต้องบังคับทุกเฟรมเอง
3. **แยกฟิลด์ตาม VergeSense** (ข้อ 2) — `person_count` / `signs_of_life` / `passively_occupied`
   เป็นคนละฟิลด์ ไม่ยุบเป็น boolean เดียว **แต่รับแค่โครงสร้าง อย่ารับ rule**:
   VergeSense นับ "laptop บนโต๊ะว่าง = occupied" ซึ่งถ้าลอกมาจะทำให้ false PRESENT
   (เกณฑ์ตัดสินหลักของเรา) พุ่งทันที
4. **Webhook on-change + REST สำหรับย้อนหลัง** (ข้อ 1) — ยิง POST เฉพาะตอน transition
   ไม่ใช่ polling ตรงกับ "แจ้งเตือน Line/Slack" ที่อยู่ใน roadmap ข้อ 14 อยู่แล้ว

### เปลี่ยนน้ำหนักการตัดสินใจ

5. **⭐ ablation ที่แพงที่สุดที่ยังไม่มีใครตอบ: crop classifier vs full-frame + center mapping**
   หลักฐานที่**หนุนสถาปัตยกรรมปัจจุบัน**ถูกตีตกทั้งหมด (0-3 ทั้งสองข้อ รวมตัวเลข 99.68%)
   ส่วนหลักฐานที่**รอด**เอียงไปทาง crop classifier (ข้อ 4, 5) — **ไม่ได้แปลว่า crop ชนะ**
   แปลว่าไม่มีใครเทียบสองทางนี้บน metric false-PRESENT เลย → ต้องวัดเองบน `desk_testset.json`
   ทำให้ roadmap ข้อ "ROI classifier แทน whole-frame detector" เลื่อนขึ้นจาก "น่าลอง" เป็น
   "ต้องวัดก่อนตัดสินใจสถาปัตยกรรมระยะยาว"
6. **อาจไม่ต้องซื้อ accelerator เลย** (ข้อ 5) — net 158K params / 0.64 MB / input 32x32
   ใช้ 0.01 วิ/zone บน Pi 5 CPU ล้วน ถ้า crop classifier ผ่าน ablation ข้อ 5 คำถาม
   Hailo/Coral/Jetson จะหมดความหมายไปเอง **ค่อยตอบทีหลัง ไม่ใช่ตอนนี้**
7. **ยืนยันคอขวดเดิมว่าถูก** (ข้อ 6) — student จิ๋ว fine-tune ด้วย pseudo-label จากกล้องหน้างาน
   ได้ 96.6-97.0% แต่**ไม่ fine-tune ได้ 81.1%** ⇒ ข้อมูลหน้างานคือคอขวด ไม่ใช่สถาปัตยกรรมโมเดล
   ตรงกับข้อสรุปใน `../context.md` ข้อ 14 ว่า "เก็บภาพโต๊ะว่างจากกล้องตัวเองก่อน คุ้มกว่ามาก"
8. **ตั้ง teacher threshold ต่ำ (0.1-0.2) ไม่ใช่สูง** (ข้อ 11) — ขัดสัญชาตญาณ แต่หลักฐานหนัก
   (445 runs) ⚠️ paper optimize mAP50 ซึ่งอาจสวนทางกับการกด false positive → ต้อง validate
   ด้วย metric ของเราเอง ไม่ใช่เชื่อเลข 0.2 ดื้อๆ
9. **PIR ต่อโต๊ะตัดทิ้งได้** (ข้อ 10) — failure mode หลักคือ false-off (คนนั่งนิ่ง = อ่านว่าว่าง)
   ซึ่งเป็นทิศทาง error ที่ผิดสำหรับโจทย์นี้พอดี

### ใช้ไม่ได้ / อย่าเชื่อ

- **ตัวเลขเทียบ sensor modality ทั้งตาราง** (กล้อง 80-96%, PIR 79-98%, ...) — ถูกตีตก 0-3
  **ไม่มีฐานเปรียบเทียบ thermal/mmWave/ToF/desk pressure เหลืออยู่เลย** ถ้าจะตัดสินใจเรื่องนี้ต้องค้นใหม่
- **ตัวเลข OpenVINO speedup 7x/3x จาก docs Ultralytics** — ถูกตีตกทั้งคู่ (0-3)
  **แต่เราวัดเองได้ 3.06x ในรอบที่ 3** ซึ่งเป็นหลักฐานที่ดีกว่า docs อยู่แล้ว → ใช้เลขของเราต่อไป
- **กฎหมาย PDPA / GDPR / works-council** — ไม่มี claim ใดรอดแม้แต่ข้อเดียว ข้อ 13 ใน `../context.md`
  ยังต้องพึ่งฝ่ายกฎหมายเหมือนเดิม งานวิจัยรอบนี้ช่วยไม่ได้
- **ตัวเลขต้นทุน USD 11/zone** — BOM ในเปเปอร์รวมได้ 120 ไม่ใช่ 177 เชื่อได้แค่ order of magnitude
  และเป็นราคา ธ.ค. 2024 (~21 เดือนก่อน)
- **หลักฐานตัวเลขเกือบทั้งหมดมาจาก parking ไม่ใช่ desk** — รถนิ่ง แยกจากพื้นหลังง่าย zone density ต่ำ
  มุมไกล ส่วนคนนั่งเก้าอี้ยากกว่าทุกแกน **โอนตัวเลขมาตรงๆ จะ optimistic เกินจริง**
