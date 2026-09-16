# data/

โฟลเดอร์เก็บข้อมูล runtime — ไม่ commit ลง git (ดู `.gitignore`)

| ไฟล์ | คำอธิบาย |
|---|---|
| `sample1.mp4` | วิดีโอ **สังเคราะห์** 960x540 (กล่องสีเทาเคลื่อนที่ฝั่งขวา) ใช้ smoke-test ว่า pipeline ทำงาน<br>ถ้าจะวัดความแม่นยำจริง ให้หาวิดีโอออฟฟิศจริงมาวางทับ เช่นจาก mixkit.co |
| `occupancy.csv` | event log — ระบบสร้างให้เองตอนรัน |
| `occupancy.sqlite` | event log (SQLite) — ระบบสร้างให้เองตอนรัน |
| `zones_calibrated.json` | ROI ที่ลากจากหน้า `/calibrate` — โหลดอัตโนมัติครั้งถัดไป |
