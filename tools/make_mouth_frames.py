#!/usr/bin/env python3
"""生成 huadun 小老虎 mouthdict 口型表 + 逐帧头部对齐偏移.

产出:
  data/avatars/<id>/mouth_frames/0.png .. 7.png   (8 档 open 家族, 张开度升序, 256x256)
  data/avatars/<id>/mouth_frames/8.png .. 13.png  (P1 viseme: wide×2/round×2/round-u/closed)
  data/avatars/<id>/mouth_frames/manifest.json    (openness 序列 + 家族→帧号映射)
  data/avatars/<id>/mouth_align.pkl               (101 帧 (dx,dy), 相对基准帧的头部位移)

口型规格采样自原画:
  口腔色 BGR≈(6,5,12)  描边色 BGR≈(26,27,29) 3-4px  下唇带浅色  无牙齿/舌
"""
import argparse, glob, json, os, pickle
import cv2
import numpy as np

MOUTH_COLOR = np.array([12, 5, 6], np.float32)     # 口腔内 (BGR, 近黑带暖)
STROKE_COLOR = (26, 27, 29)                        # 描边
LOWER_LIP = (198, 196, 210)                        # 下唇亮带
# 8 档: (开口高 h, 开口宽 w), 基准帧嘴心 (128, 150)；口腔底边=150+h，
# 最大档须 ≤200（衣领安全线 202）否则下唇被 keep_below 截断露出旧阴影带
LEVELS = [(0, 0), (7, 84), (14, 96), (24, 108), (32, 118), (39, 126), (45, 132), (50, 136)]
CX, CY_TOP = 128, 150          # 开口顶边所在行(微笑线下方)
# 衣领安全线：此行以下强制保留基准帧自身内容。基准帧衣领顶行≈206，
# 最大下唇伸到 ~201；若把基准帧衣领烘进嘴帧，贴回时头部浮动 dy(-22~3)
# 会让假衣领压在 body 真衣领上产生错位虚影（实测改动峰值 220）
BELLY_ROW = 202


def keep_below(img, base, row=BELLY_ROW):
    """裁掉绘制/擦除溢出：row 以下用基准帧像素覆盖，保证衣领随 dy 对齐"""
    img[row:, :] = base[row:, :]
    return img


def head_offsets(rois, ref_idx=87):
    """matchTemplate 测每帧相对基准帧的头部位移, 中值+EMA 平滑."""
    tmpl = rois[ref_idx][120:210, 50:210]
    dys, dxs = [], []
    for roi in rois:
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        dxs.append(loc[0] - 50); dys.append(loc[1] - 120)
    dys, dxs = np.array(dys, np.float32), np.array(dxs, np.float32)
    def smooth(a):
        k = np.array([1, 3, 5, 3, 1], np.float32); k /= k.sum()
        return np.convolve(a, k, mode='same')
    return smooth(dxs), smooth(dys)


def draw_mouth(base, h, w):
    """在基准帧副本上画一档开口: 擦除微笑线区域 → 填椭圆口腔+描边+下唇带."""
    if h <= 0:
        return base.copy()
    img = base.copy()
    cy = CY_TOP + h / 2
    axes = (int(w / 2), int(h / 2))
    center = (CX, int(cy))
    return _fill_ellipse(img, center, axes)


def _fill_ellipse(img, center, axes):
    """擦除椭圆区+6px → inpaint 融入白毛 → 画口腔/描边/下唇带."""
    h = axes[1] * 2
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, center, (axes[0] + 6, axes[1] + 6), 0, 0, 360, 255, -1)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.ellipse(img, center, axes, 0, 0, 360, STROKE_COLOR, -1)
    cv2.ellipse(img, center, axes, 0, 0, 360, MOUTH_COLOR.astype(np.uint8).tolist(), max(3, int(h * 0.10)))
    inner = (axes[0] - 4, max(axes[1] - 4, 2))
    cv2.ellipse(img, center, inner, 0, 0, 360, (6, 4, 9), -1)
    if h >= 14:
        cv2.ellipse(img, center, (axes[0] - 5, max(axes[1] - 5, 2)), 0, 20, 160, LOWER_LIP, max(3, int(h * 0.09)))
    return img


def draw_round(base, h, w):
    """圆唇 O/U: 开口收窄成近圆形, 中心下移, 两侧不碰微笑嘴角."""
    img = base.copy()
    center = (CX, CY_TOP + int(h / 2) + 4)
    return _fill_ellipse(img, center, (int(w / 2), int(h / 2)))


def draw_wide(base, h, w):
    """咧唇 EE/I: 扁宽开口. 直接覆盖绘制（不整块 inpaint 保住胡须斑点）；
    用扁椭圆而非矩形+毛色补丁圆——补丁圆与面部渐变毛色不匹配会显形。"""
    img = base.copy()
    cy = CY_TOP + 4 + max(h // 2, 4)
    axes = (w // 2, max(h // 2, 4))
    cv2.ellipse(img, (CX, cy), axes, 0, 0, 360, STROKE_COLOR, -1)
    cv2.ellipse(img, (CX, cy), (axes[0] - 3, max(axes[1] - 2, 2)), 0, 0, 360, (6, 4, 9), -1)
    if h >= 8:
        cv2.ellipse(img, (CX, cy), (axes[0] - 6, max(axes[1] - 4, 1)), 0, 25, 155,
                    LOWER_LIP, 3)
    return img


def draw_closed(base):
    """闭唇 MB: 微笑线加深加粗成抿紧黑唇 (b 前闭合/句尾收口).

    描边中心上提 2px：与基准微笑线错位，静态微笑在下、黑唇在上，
    与闭嘴档 (frame 0 = 原图) 拉开视觉差异。
    """
    img = base.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, (CX, CY_TOP + 4), (84, 9), 0, 0, 360, 255, -1)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    pts = np.array([[CX - 80, CY_TOP - 4], [CX, CY_TOP + 8], [CX + 80, CY_TOP - 4]], np.int32)
    cv2.polylines(img, [pts], False, (16, 15, 18), 9, cv2.LINE_AA)
    # 唇下淡影：增加抿嘴厚度感
    pts2 = pts + np.array([0, 7], np.int32)
    cv2.polylines(img, [pts2], False, (215, 213, 220), 3, cv2.LINE_AA)
    return img


# P1 viseme 家族帧: 序号 → (家族, 绘制函数)
# wide 宽度限制在 ±57px：右侧胡须斑点从 x=190 起（嘴心 128），描边 3px 会外溢，过宽会擦掉胡子点
VISEME_FRAMES = {
    8:  ('wide',    lambda b: draw_wide(b, 9, 108)),
    9:  ('wide',    lambda b: draw_wide(b, 13, 114)),
    10: ('round',   lambda b: draw_round(b, 30, 66)),
    11: ('round',   lambda b: draw_round(b, 46, 80)),
    12: ('round-u', lambda b: draw_round(b, 34, 46)),
    13: ('closed',  lambda b: draw_closed(b)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar-id", default="huadun")
    ap.add_argument("--ref-idx", type=int, default=87)
    args = ap.parse_args()
    root = f"data/avatars/{args.avatar_id}"
    coords = pickle.load(open(f"{root}/coords.pkl", "rb"))
    y1, y2, x1, x2 = coords[0]
    files = sorted(glob.glob(f"{root}/full_imgs/*.png"), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    rois = [cv2.imread(f)[y1:y2, x1:x2] for f in files]
    print(f"body 帧数={len(rois)} 基准帧=#{args.ref_idx}")

    dxs, dys = head_offsets(rois, args.ref_idx)
    align = [(int(round(a)), int(round(b))) for a, b in zip(dxs, dys)]
    pickle.dump({"ref_idx": args.ref_idx, "offsets": align}, open(f"{root}/mouth_align.pkl", "wb"))
    print(f"mouth_align.pkl 写出: dx∈[{min(a for a,_ in align)},{max(a for a,_ in align)}] "
          f"dy∈[{min(b for _,b in align)},{max(b for _,b in align)}]")

    out_dir = f"{root}/mouth_frames"
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(f"{out_dir}/*.png"):
        os.remove(old)
    base = rois[args.ref_idx]
    for i, (h, w) in enumerate(LEVELS):
        cv2.imwrite(f"{out_dir}/{i}.png", keep_below(draw_mouth(base, h, w), base))
    for idx, (_fam, fn) in VISEME_FRAMES.items():
        cv2.imwrite(f"{out_dir}/{idx}.png", keep_below(fn(base), base))
    manifest = {
        "openness": list(range(len(LEVELS))),
        "families": {
            "open": list(range(len(LEVELS))),
            "wide": [8, 9],
            "round": [10, 11],
            "round-u": [12],
            "closed": [13],
        },
    }
    with open(f"{out_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    n_total = len(LEVELS) + len(VISEME_FRAMES)
    print(f"口型表 {n_total} 帧（open 8 档 + viseme 6）+ manifest → {out_dir}")
    # 联排预览
    strip = np.hstack([cv2.resize(cv2.imread(f"{out_dir}/{i}.png"), (128, 128)) for i in range(n_total)])
    cv2.imwrite("/tmp/mouth_table.png", strip)
    print("预览: /tmp/mouth_table.png (前 8 格 open 张合序列, 后 6 格 wide×2/round×2/round-u/closed)")


if __name__ == "__main__":
    main()
