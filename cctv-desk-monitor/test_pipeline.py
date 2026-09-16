"""
ทดสอบ pipeline โดยฉีด detector ปลอม - ไม่ต้องมี torch/ultralytics
ตรวจ state machine, endpoints ทั้งหมด, calibrate + persist และเคสบั๊กเดิม

รัน:  python test_pipeline.py
"""
import json, os, sys, threading, time, urllib.error, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import desk_occupancy_multi as m

class FakeDetector(m.PersonDetector):
    """คืนคน 1 คนฝั่งขวาของเฟรมเสมอ (ควรตก desk-2) ; desk-1 ต้องเป็น AWAY"""
    name = "fake"
    def detect(self, frame):
        return [(0.72, 0.5, 0.9)]

cfg = m.load_config("occupancy.yaml")
for c in cfg["cameras"]:
    for z in c["zones"]:
        z["away_seconds"] = 1
events = m.EventLog("data/test_occupancy.csv", "data/test_occupancy.sqlite")
w = m.CameraWorker(0, cfg["cameras"][0], FakeDetector(), events, 480)
w.start()

app = m.build_app([w], events)
srv = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=8099,
                       threaded=True, use_reloader=False), daemon=True)
srv.start()
time.sleep(4)

def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:8099{path}", timeout=10) as r:
        return r.status, r.headers.get("Content-Type",""), r.read()

fails = []
def check(label, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("  " + extra if extra else ""))
    if not cond: fails.append(label)

print("\n--- state machine ---")
st = json.loads(get("/status")[2])
by = {z["zone_id"]: z for z in st}
check("desk-1 (ไม่มีคน) = AWAY", by["desk-1"]["state"] == "AWAY", by["desk-1"]["state"])
check("desk-2 (มีคน) = PRESENT", by["desk-2"]["state"] == "PRESENT", by["desk-2"]["state"])
check("desk-2 นับได้ 1 คน", by["desk-2"]["person_count"] == 1)
check("desk-1 นับได้ 0 คน", by["desk-1"]["person_count"] == 0)

print("\n--- endpoints ---")
s, ct, b = get("/");            check("GET /", s == 200 and b"Desk Occupancy" in b)
s, ct, b = get("/calibrate");   check("GET /calibrate", s == 200 and b"save_zones" in b)
s, ct, b = get("/summary")
sm = json.loads(b);             check("GET /summary", sm["total_zones"] == 2 and sm["PRESENT"] == 1 and sm["AWAY"] == 1, str(sm))
s, ct, b = get("/events")
ev = json.loads(b);             check("GET /events มี event", len(ev) >= 1, f"{len(ev)} events")
s, ct, b = get("/snapshot/0");  check("GET /snapshot/0 = jpeg", s == 200 and ct == "image/jpeg" and b[:2] == b"\xff\xd8", f"{len(b)} bytes")
try:
    get("/snapshot/99"); check("GET /snapshot/99 = 404", False)
except urllib.error.HTTPError as e:
    check("GET /snapshot/99 = 404", e.code == 404)

# MJPEG: อ่านแค่ chunk แรกพอ
req = urllib.request.urlopen("http://127.0.0.1:8099/video/0", timeout=10)
head = req.read(200); req.close()
check("GET /video/0 = MJPEG", b"--frame" in head and b"image/jpeg" in head)

print("\n--- POST /save_zones ---")
body = json.dumps({"cameras":[{"index":0,"id":"cam-1","zones":[
    {"id":"desk-1","name":"โต๊ะซ้ายใหม่","roi":[0,0,300,540],"away_seconds":5},
    {"id":"desk-9","name":"โต๊ะใหม่","roi":[600,100,900,500],"away_seconds":5}]}]}).encode()
r = urllib.request.Request("http://127.0.0.1:8099/save_zones", data=body,
                           headers={"Content-Type":"application/json"}, method="POST")
res = json.loads(urllib.request.urlopen(r, timeout=10).read())
check("save_zones ตอบ ok", res.get("ok") and res.get("zones") == 2, str(res))
check("ไฟล์ zones_calibrated.json ถูกเขียน", m.ZONES_FILE.exists())
saved = json.loads(m.ZONES_FILE.read_text(encoding="utf-8"))
check("persist ชื่อไทยถูกต้อง", saved["cam-1"][0]["name"] == "โต๊ะซ้ายใหม่")
time.sleep(2.5)
st2 = {z["zone_id"]: z for z in json.loads(get("/status")[2])}
check("โซนใหม่ apply สดทันที", set(st2) == {"desk-1","desk-9"}, str(set(st2)))
check("desk-9 (ครอบคนอยู่) = PRESENT", st2["desk-9"]["state"] == "PRESENT", st2["desk-9"]["state"])
check("desk-1 คงสถานะ AWAY เดิมไว้", st2["desk-1"]["state"] == "AWAY")

print("\n--- บั๊กเดิม: ส่ง dict แทน bool ---")
z = m.Zone(id="t", name="t", roi=[0,0,10,10], away_seconds=1)
try:
    z.update({"person": 1}, time.time()); check("ปฏิเสธ dict", False)
except TypeError as e:
    check("ปฏิเสธ dict (ไม่กลายเป็น PRESENT)", True, str(e)[:40])

print("\n--- apply_saved_zones ตอน restart ---")
cfg2 = m.load_config("occupancy.yaml")
m.apply_saved_zones(cfg2)
ids = [z["id"] for z in cfg2["cameras"][0]["zones"]]
check("โหลดโซนที่ calibrate ไว้กลับมา", ids == ["desk-1","desk-9"], str(ids))

w.stop_flag.set()
print("\n" + ("ทั้งหมดผ่าน" if not fails else f"ไม่ผ่าน {len(fails)}: {fails}"))
sys.exit(1 if fails else 0)
