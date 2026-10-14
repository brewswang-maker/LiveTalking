#!/usr/bin/env python3
"""
用「讲解.mp4」重建 huadun 的 body 素材（full_imgs + face_imgs + coords.pkl），
专为回退 wav2lip 方案准备。

- 讲解.mp4 身体动作明显 → 说话时能看到"讲解"手势
- bbox 用已验证过的 (255,512,511,768) 256×256 正方形（wav2lip 友好）
- coords.pkl 用 tuple 格式（旧格式，最简）

用法：
    python3 tools/rebuild_body_from_jieshao.py \
        --video /path/to/讲解.mp4 \
        --avatar-id huadun

只重写 data/avatars/<id>/{full_imgs,face_imgs,coords.pkl}，
其他 mouth_frames / eyes / expressions / breath_idle 目录保留不动。
"""
import argparse
import os
import shutil
import subprocess
import sys
import glob
import pickle

import cv2
import numpy as np

OUT_W, OUT_H = 720, 1280
FPS = 25
# wav2lip 友好 bbox：256×256 正方形，嘴巴落在裁切区的中下部（57%~75%），
# 衣领只占底部 15%（旧 bbox (255,512,511,768) 嘴巴在裁切区中部偏上、
# 衣领占底部 32%，导致神经网络把生成嘴画在衣领上→虚影+嘴不动）
BBOX = (255, 470, 511, 726)  # x1, y1, x2, y2 in 720×1280
IMG_SIZE = 256


def run(cmd):
    print("[cmd]", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:500])
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")


def extract(video, out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    vf = f"fps={FPS},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H}"
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vf", vf,
         os.path.join(out_dir, "%08d.png")])


def unsharp(img, amount=0.8, radius_sigma=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), radius_sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="讲解.mp4 绝对路径")
    ap.add_argument("--avatar-id", default="huadun")
    ap.add_argument("--bbox", default=None,
                    help="覆盖默认 bbox，格式 x1,y1,x2,y2（如 255,470,511,726）")
    args = ap.parse_args()

    global BBOX
    if args.bbox:
        BBOX = tuple(int(v) for v in args.bbox.split(","))
        assert len(BBOX) == 4, "bbox 需 4 个整数"

    if not os.path.isfile(args.video):
        sys.exit(f"✗ 视频不存在: {args.video}")

    base = f"data/avatars/{args.avatar_id}"
    os.makedirs(base, exist_ok=True)

    tmp = "/tmp/jieshao_frames"
    print(f"\n[1/4] 抽帧 {args.video} → {tmp}")
    extract(args.video, tmp)
    frames = sorted(glob.glob(os.path.join(tmp, "*.png")))
    print(f"  共 {len(frames)} 帧")
    if not frames:
        sys.exit("✗ 抽帧失败")

    full_dir = os.path.join(base, "full_imgs")
    face_dir = os.path.join(base, "face_imgs")
    for d in (full_dir, face_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    print(f"\n[2/4] 复制 full_imgs + unsharp 锐化")
    for p in frames:
        img = cv2.imread(p)
        if img is None:
            continue
        cv2.imwrite(os.path.join(full_dir, os.path.basename(p)), unsharp(img))

    print(f"\n[3/4] 按 bbox={BBOX} 裁 face_imgs 到 {IMG_SIZE}×{IMG_SIZE}")
    x1, y1, x2, y2 = BBOX
    coords = []  # tuple 格式：(y1, y2, x1, x2) 与 wav2lip paste_back_frame 一致
    for p in sorted(glob.glob(os.path.join(full_dir, "*.png"))):
        img = cv2.imread(p)
        if img is None:
            continue
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(face_dir, os.path.basename(p)), crop)
        coords.append((int(y1), int(y2), int(x1), int(x2)))

    print(f"\n[4/4] 写 coords.pkl")
    with open(os.path.join(base, "coords.pkl"), "wb") as f:
        pickle.dump(coords, f)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n✓ 完成: full_imgs={len(coords)}, face_imgs={len(coords)}, coords.pkl={len(coords)}")
    print(f"  下一步启动 wav2lip:")
    print(f"  python app.py --transport webrtc --model wav2lip --avatar_id {args.avatar_id} "
          f"--modelres 256 --customvideo_config custom.json")


if __name__ == "__main__":
    main()
