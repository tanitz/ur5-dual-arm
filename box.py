#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================================================================
 6D Pose ของกล่อง/ลัง จาก RGB + Depth  — เวอร์ชัน 2
======================================================================

ต่างจาก rgbd_box_pose.py (v1) ยังไง
--------------------------------------------------------------------
v1 ตรวจจับ "ปากลังที่เปิด" อย่างเดียว พอเจอลังปิดฝาก็ล้มทันที และปรับ
--size เท่าไรก็ไม่ช่วย เพราะมันล้มตั้งแต่ขั้นตรวจจับ ยังไม่ทันถึงขั้นใช้ขนาด

v2 แยกปัญหาออกเป็นสองชั้นให้ชัด:

    ชั้นตรวจจับ (RGB)  : หา "สี่เหลี่ยมที่น่าสนใจ" ทุกอันในภาพ ด้วยหลายกลยุทธ์
    ชั้นตัดสิน (Depth) : ให้ depth บอกว่าอันไหนคือของจริง และขนาดเท่าไร

ข้อสังเกตที่ทำให้ออกแบบแบบนี้ได้: ปากลังเปิด / ฝาลังปิด / หน้าบนกล่องที่วาง
อยู่ข้างบน — ทั้งหมดคือ "สี่เหลี่ยมบนระนาบใน 3D" เหมือนกันหมด ต่างกันแค่
วิธีหามันในภาพเท่านั้น แกน PnP + ตรวจสอบไขว้จึงใช้ร่วมกันได้ทั้งหมด

สามกลยุทธ์ตรวจจับ
--------------------------------------------------------------------
    closed : Canny -> contour ที่เป็นวงปิด
             ใช้ได้กับ "ปากลังเปิด" (ขอบสันเป็นวงปิดสมบูรณ์)

    hull   : Canny -> dilate -> convex hull -> บีบเหลือ 4 มุม
             ใช้ได้กับ "ฝาลังปิด" ที่ขอบขาดเพราะมีของวางทับ
             (กรณีจริง: กล่องขาววางคร่อมขอบฝา ทำให้ contour ไม่เป็นวงปิด
              contourArea เลยได้เกือบศูนย์ กลยุทธ์ closed จึงหาไม่เจอเลย)

    bright : แยกด้วยความสว่างหลายระดับ
             ใช้ได้กับ "วัตถุสีอ่อนบนพื้นเข้ม" เช่นกล่องกระดาษขาวบนฝาลังเทา

วิธีตัดของปลอมทิ้ง
--------------------------------------------------------------------
    1. ความตรงของสัน   : สันจริงต้องเป็นเส้นตรงใน 3D (ทดสอบจริง: ของจริง
                          2-5 mm, ของปลอมจากเก้าอี้พื้นหลัง 42.8 mm)
    2. ตรวจสอบไขว้      : RGB+PnP ทำนายระยะ vs depth วัดระยะ ต้องตรงกัน
                          (ของจริง 7-18 mm, ของปลอม 620 mm)

ติดตั้ง / วิธีรัน
--------------------------------------------------------------------
    pip install opencv-python numpy

    python box_pose_v2.py --npz scene.npz --vis out.png
    python box_pose_v2.py --npz scene.npz --sizes 600x400 195x100
"""

import argparse
import json
import sys

import cv2
import numpy as np

# ขนาดที่ใช้เป็นตัวเลือกให้ระบบเดา (ยาว x กว้าง มม.)
#
# สำคัญ: การใส่ขนาดที่ถูกต้องลงในรายการนี้คือสิ่งที่ทำให้ระบบแม่น
# ทดสอบจริงกับกล่องหน้ากากอนามัย: ถ้าให้ระบบประมาณขนาดเอง ได้ 184x75 mm
# ซึ่งผิดจนตัวกรองระนาบตัดทิ้ง (RMS 228 mm) แต่พอใส่ 195x100 ที่ถูกต้อง
# กลับได้ RMS 5.2 mm ซึ่งดีกว่าลังใหญ่เสียอีก
#
# เหตุผล: การประมาณขนาดอิสระต้องอาศัยอัตราส่วนด้านจากมุมในภาพ ซึ่งกับ
# วัตถุที่ถูกบีบแบนมาก (หน้าบนกล่องขาวสูงแค่ 25 px) จะไวต่อความคลาดของ
# มุมอย่างรุนแรง ถ้ารู้ขนาดจริงให้ใส่มาเสมอ อย่าปล่อยให้เดา
STANDARD_SIZES = [
    # ลังพลาสติก Euro/KLT
    (600, 400), (400, 300), (800, 600), (300, 200),
    (500, 300), (600, 800), (594, 396), (396, 297),
    # กล่องกระดาษขนาดเล็กที่มักวางบนลัง
    (195, 100), (200, 150), (250, 150), (150, 100), (300, 200),
]


# ======================================================================
# ชั้นที่ 1: หาสี่เหลี่ยมผู้สมัครจากภาพสี
# ======================================================================


def _order(p):
    """เรียงมุมเป็น บนซ้าย -> บนขวา -> ล่างขวา -> ล่างซ้าย"""
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p[:, 0] + p[:, 1])), axis=0)


def _valid(q, H, W, min_area):
    """คัดสี่เหลี่ยมที่เป็นไปไม่ได้ทิ้งตั้งแต่ต้น ลดงานของชั้น depth"""
    if cv2.contourArea(q.astype(np.float32)) < min_area:
        return False
    # ติดขอบภาพ = วัตถุถูกตัด มุมที่ได้จะไม่ใช่มุมจริง
    if (q[:, 0].min() < 2 or q[:, 1].min() < 2
            or q[:, 0].max() > W - 3 or q[:, 1].max() > H - 3):
        return False
    if min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)) < 20:
        return False
    # perspective ของสี่เหลี่ยมจริงไม่ควรบิดมุมเกินช่วงนี้
    for i in range(4):
        a, b = q[(i - 1) % 4] - q[i], q[(i + 1) % 4] - q[i]
        cosang = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        if not (45 < np.degrees(np.arccos(np.clip(cosang, -1, 1))) < 135):
            return False
    return True


def _hull_to_quad(contour):
    """บีบ convex hull ให้เหลือ 4 มุม โดยไล่หา eps ที่พอดี

    ต้องไล่หาเพราะ eps ที่ใช้ได้ขึ้นกับความเรียบของขอบ ซึ่งต่างกันมาก
    ระหว่างขอบลังพลาสติกคมๆ กับขอบกล่องกระดาษที่ยับ
    """
    h = cv2.convexHull(contour)
    peri = cv2.arcLength(h, True)
    for eps in np.arange(0.01, 0.10, 0.004):
        ap = cv2.approxPolyDP(h, eps * peri, True)
        if len(ap) == 4:
            return _order(ap.reshape(4, 2).astype(np.float64))
    return None


def _strategy_closed(gray, H, W, min_area):
    """ขอบที่เป็นวงปิด — ใช้ได้กับปากลังเปิด"""
    edges = cv2.Canny(cv2.bilateralFilter(gray, 9, 75, 75), 30, 90)
    out = []
    for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < min_area:
            continue
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            q = _order(ap.reshape(4, 2).astype(np.float64))
            if _valid(q, H, W, min_area):
                out.append(q)
    return out


def _strategy_hull(gray, H, W, min_area):
    """convex hull — ใช้ได้กับฝาลังที่ขอบขาดเพราะมีของวางทับ"""
    edges = cv2.Canny(cv2.bilateralFilter(gray, 9, 75, 75), 30, 90)
    out = []
    for k in (3, 5, 7):     # dilate หลายขนาดเพราะช่องว่างของขอบไม่เท่ากัน
        ed = cv2.dilate(edges, np.ones((k, k), np.uint8))
        cnts = cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            if cv2.contourArea(c) < min_area:
                continue
            q = _hull_to_quad(c)
            if q is not None and _valid(q, H, W, min_area):
                out.append(q)
    return out


def _strategy_bright(gray, H, W, min_area):
    """แยกด้วยความสว่าง — ใช้ได้กับวัตถุสีอ่อนบนพื้นเข้ม

    ต้องไล่หลายระดับเพราะหน้าบนกับหน้าข้างของกล่องเดียวกันสว่างไม่เท่ากัน
    (วัดจริง: หน้าบน 253, หน้าหน้า 180, ฝาลัง 149) ระดับต่ำจะได้ทั้งกล่อง
    ระดับสูงจะได้เฉพาะหน้าบน ซึ่งเป็นสี่เหลี่ยมบนระนาบที่เราต้องการ
    ใช้ MORPH_OPEN เบาๆ อย่างเดียว ถ้า CLOSE แรงไปหน้าบนจะเชื่อมกับหน้าหน้า
    """
    out = []
    for pct in (88, 92, 96, 98, 99):
        t = float(np.percentile(gray, pct))
        _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(th)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < min_area or x < 2 or y < 2 or x + w > W - 3 or y + h > H - 3:
                continue
            c = cv2.findContours((lab == i).astype(np.uint8) * 255,
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            q = _hull_to_quad(c)
            if q is not None and _valid(q, H, W, min_area):
                out.append(q)
    return out


def generate_candidates(color, min_area=2000):
    """รวมผลจากทุกกลยุทธ์ แล้วรวมอันที่ซ้ำกัน"""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    cands = []
    for name, fn in (("closed", _strategy_closed), ("hull", _strategy_hull),
                     ("bright", _strategy_bright)):
        for q in fn(gray, H, W, min_area):
            cands.append({"quad": q, "method": name})

    keep = []
    for c in cands:
        dup = next((k for k in keep
                    if np.max(np.linalg.norm(k["quad"] - c["quad"], axis=1)) < 20), None)
        if dup is None:
            keep.append(c)
        else:
            dup["quad"] = (dup["quad"] + c["quad"]) / 2
    return keep


# ======================================================================
# ชั้นที่ 2: depth วัดและตัดสิน
# ======================================================================


def measure_near_edge(depth, K, A, B, n_samples=90, scan=14,
                      z_min=0.3, z_max=5.0):
    """วัดสันด้านใกล้ของสี่เหลี่ยมใน 3D จาก depth

    เลือกด้านที่ใกล้กล้องที่สุดเพราะทดสอบจริงพบว่าด้านไกลมักไม่มี depth
    ที่ใช้ได้ (เป็นรอยต่อความลึก เซ็นเซอร์อ่านทะลุสันบางๆ ไปโดนพื้นหลัง)

    ที่แต่ละจุดตัวอย่าง สแกนตั้งฉากกับแนวสันแล้วเลือกค่าที่ใกล้กล้องที่สุด
    เพราะสันคือขอบบนของวัตถุ จึงเป็นผิวที่ใกล้ที่สุดในแนวสแกนนั้น
    วิธีนี้ทนต่อการที่มุมจาก RGB คลาดไปสองสามพิกเซล
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    dvec = B - A
    L = np.linalg.norm(dvec)
    if L < 1e-6:
        return None
    perp = np.array([-dvec[1], dvec[0]]) / L

    pts = []
    for t in np.linspace(0.02, 0.98, n_samples):
        base = A + dvec * t
        best = None
        for s in range(-scan, scan + 1):
            ix, iy = np.round(base + perp * s).astype(int)
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            z = depth[iy, ix]
            if z_min < z < z_max and (best is None or z < best[0]):
                best = (z, ix, iy)
        if best:
            z, ix, iy = best
            pts.append([(ix - cx) * z / fx, (iy - cy) * z / fy, z])

    if len(pts) < 20:
        return None
    P = np.array(pts)

    # ตัด outlier: จุดที่ depth หลุดจากกลุ่มหลักคือค่าที่อ่านทะลุไปพื้นหลัง
    z = P[:, 2]
    mad = np.median(np.abs(z - np.median(z))) + 1e-6
    keep = np.abs(z - np.median(z)) < 4 * mad
    if keep.sum() >= 20:
        P = P[keep]

    c = P.mean(0)
    direction = np.linalg.svd(P - c, full_matrices=False)[2][0]
    proj = (P - c) @ direction
    resid = np.linalg.norm((P - c) - np.outer(proj, direction), axis=1)
    lo, hi = np.percentile(proj, 1), np.percentile(proj, 99)
    return {"length": float(hi - lo), "center": c,
            "straightness": float(resid.std()), "n_points": len(P)}


def pose_from_quad(quad, K, L, W):
    """PnP: มุมทั้งสี่ + ขนาดที่รู้ -> 6D pose

    object frame: origin กึ่งกลางหน้าสี่เหลี่ยม, X = ด้านยาว, Y = ด้านสั้น,
                  Z = ตั้งฉากกับหน้านั้น
    ใช้ IPPE เพราะจุดทั้งสี่อยู่ระนาบเดียวกัน
    """
    dist = np.zeros(5)
    obj = np.float32([[-L/2, W/2, 0], [L/2, W/2, 0], [L/2, -W/2, 0], [-L/2, -W/2, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(np.float32), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec.ravel()
    near_mid = R @ np.array([0.0, -W/2, 0.0]) + tvec.ravel()
    return {"T": T, "rvec": rvec, "tvec": tvec,
            "reproj": float(np.linalg.norm(proj.reshape(-1, 2) - quad, axis=1).mean()),
            "near_mid_dist": float(np.linalg.norm(near_mid))}


def plane_consistency(quad, depth, K, T, shrink=0.25):
    """ตรวจว่า pose ที่ได้ทำนายค่า depth ทั่วทั้งหน้าสี่เหลี่ยมได้ถูกไหม

    ทำไมต้องมีขั้นนี้: การตรวจสอบไขว้กับขนาดมาตรฐานใช้ได้เฉพาะตอนที่วัตถุมี
    ขนาดมาตรฐาน ถ้าเป็นวัตถุขนาดอิสระ เราใช้ depth เป็นตัวกำหนดสเกลไปแล้ว
    ค่าที่ได้จึงตรงกันเองโดยอัตโนมัติ ตรวจสอบอะไรไม่ได้เลย

    ขั้นนี้ตรวจคนละอย่าง: เอา "ระนาบ" ที่ได้จาก pose ไปทำนายว่าแต่ละพิกเซล
    ในหน้าสี่เหลี่ยมควรมี depth เท่าไร แล้วเทียบกับที่เซ็นเซอร์วัดได้จริง
    การตรวจนี้ไม่ขึ้นกับสเกล จึงใช้ได้กับวัตถุขนาดอิสระด้วย

    coverage ต่ำ = ไม่มี depth บนหน้านั้นให้ตรวจ (เช่นผิวขาวสว่างจนอ่านไม่ได้)
    -> ไม่ได้แปลว่าผิด แต่แปลว่า "ยืนยันไม่ได้" ซึ่งต่างกันมาก
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    ctr = quad.mean(0)
    inner = (ctr + (quad - ctr) * (1 - shrink)).astype(np.int32)

    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [inner], 255)
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return {"coverage": 0.0, "rms_mm": None, "front_frac": 0.0,
                "behind_mm": None, "on_plane_frac": 0.0, "n": 0}

    n = T[:3, :3] @ np.array([0.0, 0.0, 1.0])      # normal ของหน้าในพิกัดกล้อง
    d0 = -n @ T[:3, 3]

    rays = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones(len(xs))], -1)
    denom = rays @ n
    ok = np.abs(denom) > 1e-6
    pred = np.full(len(xs), np.nan)
    pred[ok] = (-d0 / denom[ok]) * rays[ok, 2]

    meas = depth[ys, xs]
    valid = ok & (meas > 0.1) & np.isfinite(pred) & (pred > 0.1)
    if valid.sum() < 20:
        return {"coverage": float(valid.mean()), "rms_mm": None, "front_frac": 0.0,
                "behind_mm": None, "on_plane_frac": 0.0, "n": int(valid.sum())}

    # err > 0 = สิ่งที่วัดได้อยู่ "ลึกกว่า" ระนาบ
    # err < 0 = อยู่ "ตื้นกว่า" ระนาบ คือโผล่ออกมาข้างหน้า
    err = meas[valid] - pred[valid]

    # ต้องตรวจแบบทิศทางเดียว ไม่ใช่ RMS สองทาง เพราะ:
    #   ลังปิดฝา / หน้าบนกล่อง -> ในกรอบคือผิวนั้นเอง err ควรใกล้ 0
    #   ลังเปิด                -> ในกรอบคือก้นลังซึ่งลึกกว่าปากลัง err บวกมาก
    # ถ้าใช้ RMS สองทาง ลังเปิดจะถูกตัดทิ้งทั้งที่ถูกต้อง (ทดสอบจริง: RMS 269 mm)
    # สิ่งที่ผิดจริงคือมีของ "โผล่หน้า" ระนาบ แปลว่าระนาบวางผิดที่
    front = float(np.mean(err < -0.030))       # สัดส่วนจุดที่ตื้นกว่าระนาบเกิน 30 mm
    behind_median = float(np.median(err))

    inlier = np.abs(err) < 0.030
    rms = float(np.sqrt(np.mean(err[inlier] ** 2)) * 1000) if inlier.sum() >= 10 else None

    return {"coverage": float(valid.mean()),
            "rms_mm": rms,                      # ความแนบของจุดที่อยู่บนระนาบจริง
            "front_frac": front,                # ตัวชี้ขาดว่าระนาบผิดหรือไม่
            "behind_mm": behind_median * 1000,  # บวกมาก = เป็นภาชนะเปิด
            "on_plane_frac": float(inlier.mean()),
            "n": int(valid.sum())}


def analyse_candidate(quad, depth, K, sizes, max_disagree, max_straight):
    """ตรวจผู้สมัครหนึ่งอัน คืน dict ผลหรือ None ถ้าตกเกณฑ์"""
    edge = measure_near_edge(depth, K, quad[3], quad[2])
    if edge is None:
        return {"ok": False, "reason": "depth วัดสันไม่ได้"}
    if edge["straightness"] > max_straight:
        return {"ok": False, "reason": f"สันไม่ตรง ({edge['straightness']*1000:.1f} mm)",
                "edge": edge}

    d_meas = float(np.linalg.norm(edge["center"]))

    best = None
    for L, W in sizes:
        r = pose_from_quad(quad, K, L / 1000.0, W / 1000.0)
        if r is None:
            continue
        dis = r["near_mid_dist"] - d_meas
        if best is None or abs(dis) < abs(best["disagree"]):
            best = dict(r, L=L, W=W, disagree=dis)

    # ประมาณขนาดอิสระ: ใช้ความยาวสันที่ depth วัดได้เป็นสเกลของด้านยาว
    # แล้วกวาดหาอัตราส่วนที่ให้ reprojection ต่ำสุด
    Lm = edge["length"]
    free = None
    for ar in np.arange(1.02, 3.0, 0.01):
        r = pose_from_quad(quad, K, Lm, Lm / ar)
        if r and (free is None or r["reproj"] < free["reproj"]):
            free = dict(r, L=Lm * 1000, W=Lm / ar * 1000, aspect=ar)

    matched = best is not None and abs(best["disagree"]) <= max_disagree
    chosen = best if matched else free
    if chosen is None:
        return {"ok": False, "reason": "PnP ล้มเหลว", "edge": edge}

    pc = plane_consistency(quad, depth, K, chosen["T"])

    # ตัดของปลอม: ต้องไม่มีของโผล่ "หน้า" ระนาบมากเกินไป
    # (ของที่อยู่ลึกกว่าระนาบเป็นเรื่องปกติ นั่นคือช่องเปิดของภาชนะ)
    # เกณฑ์ต้องหลวมพอสมควร เพราะ "ของที่วางอยู่บนผิวนั้น" ก็นับเป็นจุดโผล่หน้า
    # ด้วย (กรณีจริง: กล่องขาววางบนฝาลังทำให้ 34% ของจุดโผล่หน้าระนาบฝา
    # ทั้งที่ระนาบฝาถูกต้อง) แต่ระนาบที่วางผิดที่จริงๆ จะให้ 60-73%
    if pc["n"] >= 20 and pc["front_frac"] > 0.50:
        return {"ok": False,
                "reason": f"มีพื้นผิวโผล่หน้าระนาบ {pc['front_frac']*100:.0f}% "
                          f"— ระนาบวางผิดที่"}

    # แยกลังเปิดกับลังปิดได้ฟรีจากค่านี้: ก้นลังอยู่ลึกกว่าปากลังเสมอ
    if pc["behind_mm"] is not None and pc["behind_mm"] > 40:
        container = f"ภาชนะเปิด (ก้นลึกจากปาก {pc['behind_mm']:.0f} mm)"
    elif pc["behind_mm"] is not None:
        container = "ผิวปิด/ทึบ"
    else:
        container = "ไม่ทราบ (ไม่มี depth)"

    if matched:
        confidence = "ยืนยันแล้ว"
    elif pc["on_plane_frac"] > 0.3:
        confidence = "ประมาณ (ระนาบตรงกับ depth)"
    else:
        confidence = "ยืนยันไม่ได้ (ไม่มี depth บนหน้านั้น)"

    R = chosen["T"][:3, :3]
    return {
        "ok": True, "matched_standard": matched, "confidence": confidence,
        "quad": quad,
        "pose": chosen["T"], "rvec": chosen["rvec"], "tvec": chosen["tvec"],
        "size_mm": (chosen["L"], chosen["W"]),
        "distance": float(np.linalg.norm(chosen["T"][:3, 3])),
        "tilt_deg": float(np.degrees(np.arccos(abs(R[2, 2])))),
        "reproj_px": chosen["reproj"],
        # รายงาน disagree เฉพาะตอนที่มันมีความหมาย คือตอนตรงขนาดมาตรฐาน
        # ถ้าเป็นขนาดอิสระ ค่านี้คือระยะห่างจากขนาดมาตรฐานที่ไม่เกี่ยวข้อง
        "disagree_mm": (best["disagree"] * 1000) if matched else None,
        "plane_rms_mm": pc["rms_mm"], "plane_coverage": pc["coverage"],
        "container": container, "behind_mm": pc["behind_mm"],
        "front_frac": pc["front_frac"],
        "edge_len_mm": Lm * 1000,
        "straightness_mm": edge["straightness"] * 1000,
        "n_depth_points": edge["n_points"],
    }


def detect(color, depth, K, sizes=None, max_disagree=0.030,
           max_straight=0.020, min_area=1500, verbose=True,
           show_unverified=False):
    """ตัวหลัก: หาสี่เหลี่ยมทุกอันที่เป็นวัตถุจริง เรียงตามความน่าเชื่อถือ"""
    log = print if verbose else (lambda *a, **k: None)
    sizes = sizes or STANDARD_SIZES

    cands = generate_candidates(color, min_area)
    log(f"[1] RGB หาผู้สมัครได้ {len(cands)} อัน")

    good, bad = [], []
    for c in cands:
        r = analyse_candidate(c["quad"], depth, K, sizes, max_disagree, max_straight)
        r["method"] = c["method"]
        r["area_px"] = float(cv2.contourArea(c["quad"].astype(np.float32)))
        (good if r.get("ok") else bad).append(r)

    log(f"[2] Depth ตัดสิน: ผ่าน {len(good)}, ตก {len(bad)}")
    for r in bad:
        log(f"    ตัดทิ้ง [{r['method']}] — {r['reason']}")

    # ของที่ตรงขนาดมาตรฐานน่าเชื่อถือกว่า แล้วค่อยเรียงตามความไม่ตรงกัน
    good.sort(key=lambda r: (not r["matched_standard"], r["front_frac"]))

    # ค่าตั้งต้นแสดงเฉพาะอันที่ยืนยันได้ เพราะอันที่ "ยืนยันไม่ได้" ส่วนใหญ่
    # เป็นของปลอม (เงาสะท้อน ขาเก้าอี้) ที่บังเอิญเป็นสี่เหลี่ยมในภาพ
    if not show_unverified:
        good = [r for r in good if r["matched_standard"]]
    return good


# ======================================================================
# ตัวช่วย
# ======================================================================


def matrix_to_euler_deg(T):
    R = T[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        return np.degrees([np.arctan2(R[2, 1], R[2, 2]),
                           np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])])
    return np.degrees([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])


def load_npz(path):
    d = np.load(path)
    missing = {"depth_mm", "color", "intrinsics"} - set(d.keys())
    if missing:
        sys.exit(f"[!] ไฟล์ขาด key: {missing} (มี: {list(d.keys())})")
    ki = json.loads(str(d["intrinsics"]))
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1]], float)
    return d["color"], d["depth_mm"].astype(np.float32) / 1000.0, K


def draw(color, K, dets, max_draw=4):
    img = color.copy()
    palette = [(0, 255, 0), (0, 165, 255), (255, 120, 0), (255, 0, 255)]
    for i, r in enumerate(dets[:max_draw]):
        col = palette[i % len(palette)]
        cv2.polylines(img, [r["quad"].astype(int)], True, col, 2)
        cv2.drawFrameAxes(img, K, np.zeros(5), r["rvec"], r["tvec"],
                          max(r["size_mm"]) / 1000.0 * 0.3, 2)
        L, W = r["size_mm"]
        tag = f"{L:.0f}x{W:.0f}mm {r['distance']:.2f}m"
        if not r["matched_standard"]:
            tag += " ?"
        p = r["quad"][0].astype(int)
        cv2.putText(img, tag, (p[0], p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, tag, (p[0], p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return img


def main():
    p = argparse.ArgumentParser(description="6D pose ของกล่อง/ลัง จาก RGB+depth (v2)")
    p.add_argument("--npz", required=True)
    p.add_argument("--sizes", nargs="*", default=None,
                   help="ขนาดที่คาดหวัง เช่น 600x400 195x100 (มม.)")
    p.add_argument("--tol", type=float, default=30.0, help="ยอมให้ต่างกันกี่ มม.")
    p.add_argument("--min-area", type=int, default=1500)
    p.add_argument("--vis", type=str, default=None)
    p.add_argument("--show-all", action="store_true",
                   help="แสดงอันที่ยืนยันไม่ได้ด้วย (ปกติซ่อนเพราะมักเป็นของปลอม)")
    args = p.parse_args()

    color, depth, K = load_npz(args.npz)
    print(f"[+] ภาพ {color.shape[1]}x{color.shape[0]}, depth ใช้ได้ {100*(depth>0).mean():.1f}%")
    print(f"    fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}\n")

    sizes = None
    if args.sizes:
        sizes = [tuple(float(v) for v in s.lower().split("x")) for s in args.sizes]

    dets = detect(color, depth, K, sizes=sizes, max_disagree=args.tol / 1000.0,
                  min_area=args.min_area, show_unverified=args.show_all)

    if not dets:
        print("\n[!] ไม่พบวัตถุที่ผ่านเกณฑ์")
        sys.exit(1)

    print(f"\n{'='*70}\nพบ {len(dets)} วัตถุ\n{'='*70}")
    for i, r in enumerate(dets):
        L, W = r["size_mm"]
        print(f"\n--- วัตถุที่ {i+1}  [{r['method']}]  {r['confidence']} ---")
        print(f"  ขนาด            : {L:.0f} x {W:.0f} mm")
        print(f"  ระยะกล้อง       : {r['distance']:.4f} m")
        print(f"  มุมเงยกล้อง     : {r['tilt_deg']:.2f} deg")
        print(f"  translation (m) : {r['pose'][:3,3].round(4)}")
        print(f"  RPY (deg)       : {matrix_to_euler_deg(r['pose']).round(2)}")
        dg = f"{r['disagree_mm']:+.1f} mm" if r['disagree_mm'] is not None else "n/a (ขนาดอิสระ)"
        pr = f"{r['plane_rms_mm']:.1f} mm" if r['plane_rms_mm'] is not None else "ไม่มีข้อมูล"
        print(f"  reproj          : {r['reproj_px']:.2f} px")
        print(f"  RGB vs depth    : {dg}")
        print(f"  ระนาบ vs depth  : {pr}  (ครอบคลุม {r['plane_coverage']*100:.0f}%)")
        print(f"  ความตรงของสัน   : {r['straightness_mm']:.2f} mm")
        print(f"  ลักษณะ          : {r['container']}")

    if args.vis:
        cv2.imwrite(args.vis, draw(color, K, dets))
        print(f"\n[+] บันทึกภาพ: {args.vis}")


if __name__ == "__main__":
    main()