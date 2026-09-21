# UR5 world/base calibration

โฟลเดอร์นี้รวมขั้นตอนคาลิเบรตฐานแขน B เทียบกับฐานแขน A โดยให้แขนทั้งสอง
จับวัตถุแข็งชิ้นเดียวกัน ฐาน A ยังคงเป็นจุดอ้างอิงของ `world` และผลการวัดจะ
แก้ `arms.B.base` ใน `config/cell.yaml`

## ไฟล์

- `world_base.py` โปรแกรมเก็บ pose คำนวณ และ apply ผล
- `output/` ผลของการรันแต่ละ session
- `../ur5dual/geometry/calibration.py` ตัวแก้สมการ AX=ZB
- `../config/cell.yaml` configuration ที่ใช้งานจริง

## วิธีรัน

วัดและสร้างรายงานโดยไม่แก้ configuration:

```bash
cd /home/jetson/UR5
python3 calibrate/world_base.py
```

วัดและเขียนผลลง `config/cell.yaml` เมื่อข้อมูลผ่านเกณฑ์:

```bash
cd /home/jetson/UR5
python3 calibrate/world_base.py --apply
```

ทั้งสองแขนจะเข้า Freedrive ระหว่างเก็บข้อมูล ต้องประคองน้ำหนักวัตถุและห้าม
ให้ gripper เลื่อนตำแหน่งบนวัตถุ เก็บอย่างน้อย 8 placements โดยเลื่อนวัตถุ
และหมุนรอบอย่างน้อยสองแกนที่ไม่ขนานกัน เมื่อครบแล้วพิมพ์ `q` และกด Enter

## Output

ทุก session ที่คำนวณได้จะสร้างโฟลเดอร์:

```text
calibrate/output/world_base_YYYYMMDD_HHMMSS_microseconds/
├── cell_before.yaml
├── result.json
└── cell_after.yaml       # มีเฉพาะเมื่อ apply สำเร็จ
```

`result.json` เก็บ TCP pose ของ A และ B ทุก placement, transform
`T_baseA_baseB`, ค่าความคลาดเคลื่อน และสถานะของ session ส่วนไฟล์ที่ระบบใช้
จริงหลัง apply ยังคงเป็น `../config/cell.yaml`

สถานะสำคัญใน `result.json`:

- `report_only` วัดอย่างเดียว ไม่มีการแก้ configuration
- `apply_refused` ข้อมูลไม่ผ่านและไม่ได้แก้ configuration
- `applied_orientation_only` แก้เฉพาะมุมของฐาน B
- `applied_full` แก้ตำแหน่งและมุม และผ่านเกณฑ์ full calibration
- `applied_full_not_trusted` เขียนตำแหน่งและมุมแล้ว แต่ residual ยังไม่ผ่าน
  เกณฑ์สำหรับตั้ง `calibrated: true`

คำสั่งเดิม `python3 tests/check_hold_online.py` ยังใช้งานได้และจะเรียกไฟล์
ในโฟลเดอร์นี้ต่อให้โดยอัตโนมัติ
