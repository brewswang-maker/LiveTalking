#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从美术交付的 8 段动作视频中，自动抽取 mouthdict 四层协同所需的全部静态素材。

输出目录结构:
  data/avatars/huadun/
    mouth_frames/0.png ~ N.png   — 口型 viseme 表 (256×256 嘴区 crop)
    eyes/open.png, closed.png, left.png, right.png, down.png — 眼部 crop
    expressions/neutral.png, happy.png, surprised.png, thinking.png — 全身情绪帧
    breath_idle/nod.png, lean.png — 呼吸/点头 idle 变体
    full_imgs/ (重建) — 说话主循环帧 (720×1280)
    face_imgs/ (重建) — 对应 mouth crop
    coords.pkl — [(y1,y2,x1,x2), {"mouth":..., "eyes":...}]

  data/avatars/huadun_idle/ — audiotype=1 待机帧
  data/avatars/huadun_salute/ — audiotype=2 敬礼帧
  data/avatars/huadun_salute.wav — 等长静音轨
  custom.json — 合并注册

用法:
  python tools/build_huadun_assets.py
  python tools/build_huadun_assets.py --video_dir "/Users/mac1234/项目/数字人/只有logo"
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys

import cv2
import numpy as np

# ==================== 常量 ====================
OUT_W, OUT_H = 720, 1280
FPS = 25
MOUTH_BBOX = (300, 600, 460, 670)  # x1, y1, x2, y2 in 720×1280 — 精确覆盖嘴唇+口腔
# Eyes region: 上半脸眼睛区域
EYES_BBOX = (265, 465, 495, 555)   # x1, y1, x2, y2
IMG_SIZE = 256  # mouth_frames 目标尺寸

VIDEO_DIR_DEFAULT = "/Users/mac1234/项目/数字人/只有logo"
AVATAR_BASE = "data/avatars"
AVATAR_ID = "huadun"


def run(cmd, capture=False):
    print("[cmd]", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0:
        print(result.stderr[:500] if result.stderr else "")
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return result.stdout if capture else ""


def extract_frames_raw(video, out_dir, fps=FPS, w=OUT_W, h=OUT_H, ss=None, to=None):
    """从视频抽帧，统一分辨率和 fps。"""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    vf = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if to is not None:
        cmd += ["-to", str(to)]
    cmd += ["-i", video, "-vf", vf, os.path.join(out_dir, "%08d.png")]
    run(cmd)


def list_frames(d):
    """按帧号排序列出目录内 png。"""
    files = glob.glob(os.path.join(d, "*.png"))
    return sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


def mouth_openness_score(img, bbox=MOUTH_BBOX):
    """
    计算嘴区张开度指标：下半 face crop 中暗区占比。
    卡通角色嘴张开 = 露出深色口腔内部。
    """
    x1, y1, x2, y2 = bbox
    # 只取下半区（嘴部所在）
    mid_y = (y1 + y2) // 2
    crop = img[mid_y:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 暗区阈值: 口腔内部通常 < 80
    dark_ratio = np.sum(gray < 80) / gray.size
    return float(dark_ratio)


def select_mouth_levels(frames_dir, bbox=MOUTH_BBOX, n_levels=8):
    """
    从帧序列中按嘴张开度排序，等间距选 n_levels 个代表性档位。
    过滤掉手遮挡帧（检测 bbox 下半区边缘异常）。
    """
    files = list_frames(frames_dir)
    if not files:
        return []

    scores = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        s = mouth_openness_score(img, bbox)
        scores.append((s, f, img))

    # 按 score 排序
    scores.sort(key=lambda x: x[0])

    # 去重：score 差距 < 0.02 视为同档
    unique = []
    for s, f, img in scores:
        if not unique or s - unique[-1][0] > 0.02:
            unique.append((s, f, img))

    # 等间距采样 n_levels 档
    if len(unique) <= n_levels:
        selected = unique
    else:
        indices = np.linspace(0, len(unique) - 1, n_levels, dtype=int)
        selected = [unique[i] for i in indices]

    return selected  # [(score, filepath, image)]


def crop_mouth_patch(img, bbox=MOUTH_BBOX, size=IMG_SIZE):
    """按 bbox 裁嘴区 → resize 到 size×size。"""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def crop_eyes_patch(img, bbox=EYES_BBOX, size=IMG_SIZE):
    """按 eyes bbox 裁眼区 → resize。"""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def detect_eye_state(img, bbox=EYES_BBOX):
    """
    简易判断眼区状态：open/closed。
    闭眼时眼区暗区行占比高（一条线），睁眼时有瞳孔亮点。
    """
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 闭眼特征：高对比度水平线（眼睑边缘），整体亮度均匀
    # 睁眼特征：有明显暗区（瞳孔）+ 亮区（眼白）
    dark_ratio = np.sum(gray < 60) / gray.size
    bright_ratio = np.sum(gray > 200) / gray.size
    # 卡通角色闭眼时瞳孔消失，亮暗对比下降
    contrast = float(gray.std())
    if contrast < 35:  # 低对比度 → 闭眼
        return "closed"
    return "open"


def find_eye_frames(frames_dir, eyes_bbox=EYES_BBOX):
    """
    从帧目录中找 open 和 closed 眼帧。
    返回 (open_path, closed_path)。
    """
    files = list_frames(frames_dir)
    open_candidates = []
    closed_candidates = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        state = detect_eye_state(img, eyes_bbox)
        x1, y1, x2, y2 = eyes_bbox
        crop = img[y1:y2, x1:x2]
        contrast = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())
        if state == "closed":
            closed_candidates.append((contrast, f))
        else:
            open_candidates.append((-contrast, f))  # 对比度越高越好

    open_path = min(open_candidates, key=lambda x: x[0])[1] if open_candidates else None
    closed_path = min(closed_candidates, key=lambda x: x[0])[1] if closed_candidates else None
    return open_path, closed_path


def find_gaze_frames(frames_dir, eyes_bbox=EYES_BBOX, n=6):
    """
    从帧序列中找视线方向变化最大的几帧。
    用瞳孔位置（眼区最暗像素的 x 坐标）判断左右/上下偏移。
    """
    files = list_frames(frames_dir)
    samples = []
    step = max(1, len(files) // n) if files else 1
    for i in range(0, len(files), step):
        img = cv2.imread(files[i])
        if img is None:
            continue
        x1, y1, x2, y2 = eyes_bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 瞳孔 = 最暗 5% 像素的质心
        threshold = np.percentile(gray, 5)
        mask = (gray <= threshold).astype(np.uint8)
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = (x2 - x1) / 2, (y2 - y1) / 2
        samples.append((cx, cy, files[i]))

    if not samples:
        return {}

    # 按 cx 排序 → left/right
    samples_by_x = sorted(samples, key=lambda s: s[0])
    # 按 cy 排序 → up/down
    samples_by_y = sorted(samples, key=lambda s: s[1])

    result = {}
    result["left"] = samples_by_x[0][2]    # 瞳孔最左
    result["right"] = samples_by_x[-1][2]  # 瞳孔最右
    result["up"] = samples_by_y[0][2]      # 瞳孔最上
    result["down"] = samples_by_y[-1][2]   # 瞳孔最下
    result["center"] = samples_by_x[len(samples_by_x) // 2][2]  # 中间
    return result


def select_expression_frame(frames_dir, target="neutral"):
    """
    从帧目录选一张代表性情绪帧（全身 720×1280）。
    简化策略：取中位帧（最稳定）。
    """
    files = list_frames(frames_dir)
    if not files:
        return None
    # 取中间帧
    mid = files[len(files) // 2]
    return mid


def generate_silence_wav(out_path, duration_s, sr=16000):
    """生成等长静音 wav。"""
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono",
        "-t", f"{duration_s:.3f}",
        "-acodec", "pcm_s16le", out_path,
    ])


def update_custom_json(entries, path="custom.json"):
    """合并写入 custom.json。"""
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    for e in entries:
        existing = [x for x in existing if x.get("audiotype") != e["audiotype"]]
        existing.append(e)
    existing.sort(key=lambda c: c.get("audiotype", 0))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 更新 {path}")


def unsharp(img, amount=1.0, sigma=1.0):
    """Unsharp mask 锐化（与之前全素材一致）。"""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser(description="从 8 段视频抽取 mouthdict 四层协同素材")
    ap.add_argument("--video_dir", default=VIDEO_DIR_DEFAULT)
    ap.add_argument("--avatar_id", default=AVATAR_ID)
    ap.add_argument("--mouth_levels", type=int, default=10, help="口型档位数量 (8-12)")
    ap.add_argument("--skip_actions", action="store_true", help="跳过敬礼/待机处理")
    args = ap.parse_args()

    vd = args.video_dir
    base_dir = os.path.join(AVATAR_BASE, args.avatar_id)
    tmp_dir = "/tmp/huadun_extract"
    os.makedirs(tmp_dir, exist_ok=True)

    # 定义视频路径
    videos = {
        "jieshao": os.path.join(vd, "讲解.mp4"),
        "wufa": os.path.join(vd, "无法回答.mp4"),
        "jingli": os.path.join(vd, "敬礼.mp4"),
        "sikao": os.path.join(vd, "思考、或听.mp4"),
        "daiji": os.path.join(vd, "待机.mp4"),
        "zuixing2": os.path.join(vd, "嘴型2.mp4"),
        "ting": os.path.join(vd, "听.mp4"),
        "kouxing": os.path.join(vd, "口型.mp4"),
    }
    for k, v in videos.items():
        if not os.path.isfile(v):
            sys.exit(f"视频不存在: {v}")

    print("=" * 60)
    print("阶段 1: 抽取视频帧到临时目录")
    print("=" * 60)

    # 嘴型2.mp4 — 主口型源（无手遮挡，头位稳定）
    tmp_mouth2 = os.path.join(tmp_dir, "zuixing2")
    extract_frames_raw(videos["zuixing2"], tmp_mouth2, fps=25)
    print(f"  嘴型2: {len(list_frames(tmp_mouth2))} frames")

    # 口型.mp4 — 补充口型 + 眼帧
    tmp_kouxing = os.path.join(tmp_dir, "kouxing")
    extract_frames_raw(videos["kouxing"], tmp_kouxing, fps=25)
    print(f"  口型: {len(list_frames(tmp_kouxing))} frames")

    # 听.mp4 — 聆听态
    tmp_ting = os.path.join(tmp_dir, "ting")
    extract_frames_raw(videos["ting"], tmp_ting, fps=25)
    print(f"  听: {len(list_frames(tmp_ting))} frames")

    # 思考、或听.mp4 — 思考/视线方向
    tmp_sikao = os.path.join(tmp_dir, "sikao")
    extract_frames_raw(videos["sikao"], tmp_sikao, fps=25)
    print(f"  思考或听: {len(list_frames(tmp_sikao))} frames")

    # 无法回答.mp4 — 表情
    tmp_wufa = os.path.join(tmp_dir, "wufa")
    extract_frames_raw(videos["wufa"], tmp_wufa, fps=25)
    print(f"  无法回答: {len(list_frames(tmp_wufa))} frames")

    # 讲解.mp4 — 说话主循环
    tmp_jieshao = os.path.join(tmp_dir, "jieshao")
    extract_frames_raw(videos["jieshao"], tmp_jieshao, fps=25)
    print(f"  讲解: {len(list_frames(tmp_jieshao))} frames")

    # 待机.mp4 — idle
    tmp_daiji = os.path.join(tmp_dir, "daiji")
    extract_frames_raw(videos["daiji"], tmp_daiji, fps=25)
    print(f"  待机: {len(list_frames(tmp_daiji))} frames")

    print("\n" + "=" * 60)
    print("阶段 2: 口型 viseme 表 → mouth_frames/")
    print("=" * 60)

    mouth_out = os.path.join(base_dir, "mouth_frames")
    os.makedirs(mouth_out, exist_ok=True)

    # 从嘴型2 选取 n_levels 个不同张开度
    selected = select_mouth_levels(tmp_mouth2, MOUTH_BBOX, n_levels=args.mouth_levels)
    print(f"  嘴型2 选出 {len(selected)} 档 (score 范围 {selected[0][0]:.3f}~{selected[-1][0]:.3f})")

    level = 0
    for score, fpath, img in selected:
        patch = crop_mouth_patch(img, MOUTH_BBOX, IMG_SIZE)
        patch = unsharp(patch, amount=0.5, sigma=1.0)
        out_name = f"{level}.png"
        cv2.imwrite(os.path.join(mouth_out, out_name), patch)
        print(f"    level {level}: score={score:.4f} ← {os.path.basename(fpath)}")
        level += 1

    # 从口型.mp4 补充（如果嘴型2 覆盖不够）
    if level < args.mouth_levels:
        extra = select_mouth_levels(tmp_kouxing, MOUTH_BBOX, n_levels=args.mouth_levels - level)
        for score, fpath, img in extra:
            patch = crop_mouth_patch(img, MOUTH_BBOX, IMG_SIZE)
            patch = unsharp(patch, amount=0.5, sigma=1.0)
            cv2.imwrite(os.path.join(mouth_out, f"{level}.png"), patch)
            print(f"    level {level} (补充): score={score:.4f} ← {os.path.basename(fpath)}")
            level += 1

    print(f"  ✓ mouth_frames/: {level} 档口型")

    print("\n" + "=" * 60)
    print("阶段 3: 眼帧 → eyes/")
    print("=" * 60)

    eyes_out = os.path.join(base_dir, "eyes")
    os.makedirs(eyes_out, exist_ok=True)

    # 从口型.mp4 找 open/closed（它有明确的闭眼帧）
    open_path, closed_path = find_eye_frames(tmp_kouxing, EYES_BBOX)
    # 备选：从思考、或听 找（也有闭眼帧）
    if closed_path is None:
        open_path2, closed_path2 = find_eye_frames(tmp_sikao, EYES_BBOX)
        if closed_path2:
            closed_path = closed_path2
    if open_path is None:
        # fallback: 从待机取第一帧（肯定睁眼）
        files = list_frames(tmp_daiji)
        open_path = files[0] if files else None

    if open_path:
        img = cv2.imread(open_path)
        patch = crop_eyes_patch(img, EYES_BBOX, IMG_SIZE)
        if patch is not None:
            cv2.imwrite(os.path.join(eyes_out, "open.png"), unsharp(patch, 0.5))
            print(f"  ✓ open.png ← {os.path.basename(open_path)}")

    if closed_path:
        img = cv2.imread(closed_path)
        patch = crop_eyes_patch(img, EYES_BBOX, IMG_SIZE)
        if patch is not None:
            cv2.imwrite(os.path.join(eyes_out, "closed.png"), unsharp(patch, 0.5))
            print(f"  ✓ closed.png ← {os.path.basename(closed_path)}")

    # 视线方向：从 思考、或听 提取（有明显头部偏转）
    gaze_files = find_gaze_frames(tmp_sikao, EYES_BBOX)
    for direction, fpath in gaze_files.items():
        if fpath and os.path.isfile(fpath):
            img = cv2.imread(fpath)
            patch = crop_eyes_patch(img, EYES_BBOX, IMG_SIZE)
            if patch is not None:
                cv2.imwrite(os.path.join(eyes_out, f"{direction}.png"), unsharp(patch, 0.5))
                print(f"  ✓ {direction}.png ← {os.path.basename(fpath)}")

    # 如果缺 left/right，从听.mp4 补充
    for d in ["left", "right"]:
        if not os.path.exists(os.path.join(eyes_out, f"{d}.png")):
            gaze2 = find_gaze_frames(tmp_ting, EYES_BBOX)
            if d in gaze2 and gaze2[d]:
                img = cv2.imread(gaze2[d])
                patch = crop_eyes_patch(img, EYES_BBOX, IMG_SIZE)
                if patch is not None:
                    cv2.imwrite(os.path.join(eyes_out, f"{d}.png"), unsharp(patch, 0.5))
                    print(f"  ✓ {d}.png (补充自听.mp4)")

    print(f"  ✓ eyes/: {len(os.listdir(eyes_out))} 文件")

    print("\n" + "=" * 60)
    print("阶段 4: 情绪图集 → expressions/")
    print("=" * 60)

    expr_out = os.path.join(base_dir, "expressions")
    os.makedirs(expr_out, exist_ok=True)

    # neutral: 待机中位帧（闭嘴、正面、平静）
    neutral_src = select_expression_frame(tmp_daiji)
    if neutral_src:
        img = cv2.imread(neutral_src)
        cv2.imwrite(os.path.join(expr_out, "neutral.png"), unsharp(img, 0.8))
        print(f"  ✓ neutral.png ← 待机 中位帧")

    # happy: 讲解 第一帧（微笑、招手）
    happy_files = list_frames(tmp_jieshao)
    if happy_files:
        img = cv2.imread(happy_files[0])
        cv2.imwrite(os.path.join(expr_out, "happy.png"), unsharp(img, 0.8))
        print(f"  ✓ happy.png ← 讲解 首帧")

    # surprised: 无法回答 中前段（眉毛上扬帧）
    wufa_files = list_frames(tmp_wufa)
    if len(wufa_files) > 3:
        # 眉毛上扬大约在前 1/3
        img = cv2.imread(wufa_files[len(wufa_files) // 4])
        cv2.imwrite(os.path.join(expr_out, "surprised.png"), unsharp(img, 0.8))
        print(f"  ✓ surprised.png ← 无法回答 前1/4")

    # thinking: 思考、或听 后段（歪头、视线下）
    sikao_files = list_frames(tmp_sikao)
    if len(sikao_files) > 4:
        img = cv2.imread(sikao_files[int(len(sikao_files) * 0.7)])
        cv2.imwrite(os.path.join(expr_out, "thinking.png"), unsharp(img, 0.8))
        print(f"  ✓ thinking.png ← 思考或听 70%处")

    # listening: 听.mp4 中位帧
    listening_src = select_expression_frame(tmp_ting)
    if listening_src:
        img = cv2.imread(listening_src)
        cv2.imwrite(os.path.join(expr_out, "listening.png"), unsharp(img, 0.8))
        print(f"  ✓ listening.png ← 听 中位帧")

    print(f"  ✓ expressions/: {len(os.listdir(expr_out))} 文件")

    print("\n" + "=" * 60)
    print("阶段 5: 呼吸/点头 idle → breath_idle/")
    print("=" * 60)

    breath_out = os.path.join(base_dir, "breath_idle")
    os.makedirs(breath_out, exist_ok=True)

    # nod: 思考或听 中帧间差异最大的（头位上下移动）
    sikao_files = list_frames(tmp_sikao)
    if len(sikao_files) >= 4:
        # 取一个头略微低下的帧（nod 中间态）
        nod_img = cv2.imread(sikao_files[len(sikao_files) // 3])
        cv2.imwrite(os.path.join(breath_out, "nod.png"), unsharp(nod_img, 0.8))
        print(f"  ✓ nod.png")

    # lean: 思考或听 后段（明显侧倾）
    if len(sikao_files) >= 6:
        lean_img = cv2.imread(sikao_files[int(len(sikao_files) * 0.85)])
        cv2.imwrite(os.path.join(breath_out, "lean.png"), unsharp(lean_img, 0.8))
        print(f"  ✓ lean.png")

    # 从听.mp4 补充微侧头
    ting_files = list_frames(tmp_ting)
    if len(ting_files) >= 4:
        tilt_img = cv2.imread(ting_files[len(ting_files) // 2])
        cv2.imwrite(os.path.join(breath_out, "tilt.png"), unsharp(tilt_img, 0.8))
        print(f"  ✓ tilt.png")

    print(f"  ✓ breath_idle/: {len(os.listdir(breath_out))} 文件")

    print("\n" + "=" * 60)
    print("阶段 6: 敬礼.mp4 → audiotype=2")
    print("=" * 60)

    if not args.skip_actions:
        salute_dir = os.path.join(AVATAR_BASE, f"{args.avatar_id}_salute")
        salute_wav = os.path.join(AVATAR_BASE, f"{args.avatar_id}_salute.wav")
        # 整段提取（4s 全保留）
        extract_frames_raw(videos["jingli"], salute_dir, fps=FPS)
        n_salute = len(list_frames(salute_dir))
        dur_salute = n_salute / FPS
        generate_silence_wav(salute_wav, dur_salute)
        print(f"  ✓ {salute_dir}: {n_salute} frames, {dur_salute:.2f}s")

        print("\n" + "=" * 60)
        print("阶段 7: 待机.mp4 → audiotype=1")
        print("=" * 60)

        idle_dir = os.path.join(AVATAR_BASE, f"{args.avatar_id}_idle")
        extract_frames_raw(videos["daiji"], idle_dir, fps=FPS)
        n_idle = len(list_frames(idle_dir))
        print(f"  ✓ {idle_dir}: {n_idle} frames")

        # 注册 custom.json
        update_custom_json([
            {"audiotype": 1, "imgpath": idle_dir},
            {"audiotype": 2, "imgpath": salute_dir, "audiopath": salute_wav},
        ])

    print("\n" + "=" * 60)
    print("阶段 8: 重建 full_imgs + face_imgs + coords.pkl")
    print("=" * 60)

    # 用嘴型2.mp4 重建说话主循环（头位最稳定、无手遮挡）
    full_dir = os.path.join(base_dir, "full_imgs")
    face_dir = os.path.join(base_dir, "face_imgs")

    # 复制嘴型2 的帧作为 full_imgs（25fps, 720×1280）
    if os.path.isdir(full_dir):
        shutil.rmtree(full_dir)
    shutil.copytree(tmp_mouth2, full_dir)
    n_full = len(list_frames(full_dir))
    print(f"  full_imgs: {n_full} frames (源: 嘴型2.mp4)")

    # 按 bbox 裁 face_imgs
    if os.path.isdir(face_dir):
        shutil.rmtree(face_dir)
    os.makedirs(face_dir, exist_ok=True)
    x1, y1, x2, y2 = MOUTH_BBOX
    coords = []
    for p in list_frames(full_dir):
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        crop = img[cy1:cy2, cx1:cx2]
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(face_dir, os.path.basename(p)), crop)
        coords.append({"mouth": (int(cy1), int(cy2), int(cx1), int(cx2)),
                       "eyes": (int(EYES_BBOX[1]), int(EYES_BBOX[3]), int(EYES_BBOX[0]), int(EYES_BBOX[2]))})

    # 写 coords.pkl（新格式：dict list，兼容旧代码）
    coords_path = os.path.join(base_dir, "coords.pkl")
    with open(coords_path, "wb") as f:
        pickle.dump(coords, f)
    print(f"  ✓ coords.pkl: {len(coords)} entries (mouth + eyes bbox)")

    # 锐化 full_imgs
    for p in list_frames(full_dir):
        img = cv2.imread(p)
        if img is not None:
            cv2.imwrite(p, unsharp(img, 0.8, 1.0))
    print(f"  ✓ full_imgs 锐化完成")

    print("\n" + "=" * 60)
    print("阶段 9: 生成 asset_manifest.md")
    print("=" * 60)

    manifest = generate_manifest(base_dir, args.avatar_id)
    manifest_path = os.path.join(base_dir, "asset_manifest.md")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest)
    print(f"  ✓ {manifest_path}")

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print(f"\n启动命令:")
    print(f"  python app.py --model mouthdict --avatar_id {args.avatar_id} \\")
    print(f"    --mouth_frames data/avatars/{args.avatar_id}/mouth_frames \\")
    print(f"    --eye_frames_dir data/avatars/{args.avatar_id}/eyes \\")
    print(f"    --expression_dir data/avatars/{args.avatar_id}/expressions \\")
    print(f"    --enable_blink 1 --enable_gaze 1 --enable_breath 1 --enable_emotion 1 \\")
    print(f"    --modelres 256 --customvideo_config custom.json")


def generate_manifest(base_dir, avatar_id):
    """生成素材清单。"""
    lines = [f"# Asset Manifest — {avatar_id}\n"]
    lines.append(f"生成时间: {__import__('datetime').datetime.now().isoformat()}\n")

    sections = [
        ("mouth_frames", "口型 viseme 表", "256×256 嘴区 crop，按张开度升序编号"),
        ("eyes", "眼部图集", "眼区 crop，open/closed/方向"),
        ("expressions", "情绪图集", "全身 720×1280，按情绪命名"),
        ("breath_idle", "呼吸/点头 idle", "全身 720×1280"),
        ("full_imgs", "说话主循环帧", "720×1280 全身帧"),
        ("face_imgs", "对应嘴区裁剪", "256×256"),
    ]
    for dirname, title, desc in sections:
        d = os.path.join(base_dir, dirname)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            lines.append(f"\n## {title} (`{dirname}/`)\n")
            lines.append(f"规格: {desc}\n")
            lines.append(f"数量: {len(files)} 文件\n\n")
            for f in files[:20]:
                lines.append(f"- `{f}`\n")
            if len(files) > 20:
                lines.append(f"- ... 共 {len(files)} 个\n")

    # coords.pkl
    lines.append(f"\n## coords.pkl\n")
    lines.append(f"格式: `list[dict]`，每项 `{{'mouth': (y1,y2,x1,x2), 'eyes': (y1,y2,x1,x2)}}`\n")
    lines.append(f"mouth bbox: {MOUTH_BBOX}\n")
    lines.append(f"eyes bbox: {EYES_BBOX}\n")

    # 缺口说明
    lines.append(f"\n## 缺口/限制\n")
    lines.append("""
- **O/U 圆唇 viseme**: 素材中嘴形以"微笑开合"为主，未出现明确的圆唇（O/U）或横展（EE/I）形态。
  当前档位按"张开度"排序，缺少**唇形维度**。建议美术补拍正面静态：
  - 圆唇 O（嘴唇收拢成圆形）
  - 扁唇 EE（嘴角向两侧拉开）
  - MBP（双唇紧闭、微向前嘟）
- **眼帧方向精度**: 卡通角色瞳孔变化幅度小，left/right 帧差异可能 <5px，实际效果依赖 GazeScheduler 的 warpAffine 偏移辅助。
- **表情帧头位**: surprised/thinking 帧有轻微头位偏移（±3°），与 full_imgs 叠加时 EmotionMood 的 addWeighted 融合可容忍。
""")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
