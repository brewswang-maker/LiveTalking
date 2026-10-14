###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  查表驱动卡通嘴部动画 — 用"音频 RMS 能量 → 嘴帧档位"替代 wav2lip 神经嘴部重绘
#
#  设计动机：wav2lip/MuseTalk 基于真人训练，对卡通脸必糊；改用预渲染口型帧库 +
#  音量驱动选帧（Live2D VTuber 同款思路），嘴部为原生渲染画质，且零 GPU 推理。
#
#  数据流：
#    base_avatar.inference() 循环取到 audio_frames（每帧 2 个 20ms PCM chunk）
#      → inference_batch(index, mel, audio_frames)  【本类覆盖，跳过模型】
#          1. 每帧 RMS = sqrt(mean(concat(chunk0, chunk1)^2))
#          2. mapper(rms) → level ∈ [0, N-1]
#          3. pred[i] = mouth_lib.get(level)        # [256,256,3] 嘴部贴片
#      → res_frame_queue ← (pred[i], audio_frames_pair, idx)
#    base_avatar.process_frames() 说话分支
#      → paste_back_frame(pred[i], idx)             【复用 wav2lip 同款 bbox 贴回】
#
#  注意：骨架阶段保留 MelASR 以零改主循环节拍；若后续不需要 mel，可换轻量"仅切帧"ASR。
#

import numpy as np
from numpy.typing import NDArray

import os
import cv2
import glob
import pickle
import time

from avatars.audio_features.mel import MelASR
from avatars.base_avatar import BaseAvatar

from utils.logger import logger
from utils.image import read_imgs, mirror_index
from registry import register

# 四层协同子包（P0 骨架：默认全关时与旧版逐帧等价）
from avatars.mouthdict_layers import (
    LayerCompositor,
    BlinkScheduler,
    GazeScheduler,
    BreathScheduler,
    EmotionMood,
    VisemeTimeline,
    OcclusionDetector,
)


class MouthFrameLibrary:
    """嘴帧库加载器。

    从 data/avatars/<id>/mouth_frames/ 读取嘴帧（按文件名整数升序）。
    支持可选 manifest.json（P1 viseme 多家族）：
        {"openness": [0..7],                 # RMS 张开度序列（open 家族）
         "families": {"wide":[8,9], "round":[10,11], "round-u":[12], "closed":[13]}}
    无 manifest → 全部帧即 openness 序列（旧行为逐帧等价）。
    """

    def __init__(self, frames_dir: str, expected_size=(256, 256)):
        self.frames_dir = frames_dir
        self.expected_size = expected_size          # (W, H)
        self.levels: list[NDArray[np.uint8]] = []   # 每项 [H, W, 3] BGR，按文件名序号索引
        self.openness: list[int] = list(range(len(self.levels)))  # load() 后重建
        self.families: dict[str, list[int]] = {}    # 家族 → 帧序号升序

    def load(self) -> "MouthFrameLibrary":
        """读盘 → 排序 → 尺寸归一 → 解析 manifest。返回 self 便于链式调用。"""
        input_list = glob.glob(os.path.join(self.frames_dir, '*.[jpJP][pnPN]*[gG]'))
        # 只收整数文件名（排除 manifest.json 等同目录非帧文件）
        input_list = [p for p in input_list if os.path.splitext(os.path.basename(p))[0].isdigit()]
        input_list = sorted(input_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not input_list:
            raise FileNotFoundError(f"mouth_frames 目录为空: {self.frames_dir}")

        raw = read_imgs(input_list)
        w, h = self.expected_size
        for img in raw:
            if img is None:
                # TODO: 读图失败的具体路径已在 read_imgs 内部吞掉，必要时在此记录
                continue
            if (img.shape[1], img.shape[0]) != (w, h):
                # TODO: 尺寸不一致时是否允许，取决于美术交付规范；骨架先 resize 兜底
                img = cv2.resize(img, (w, h))
            self.levels.append(img)

        # manifest（可选）：家族 → 帧序号列表
        manifest_path = os.path.join(self.frames_dir, 'manifest.json')
        if os.path.isfile(manifest_path):
            try:
                import json
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    mf = json.load(f)
                self.openness = [i for i in mf.get('openness', []) if 0 <= i < len(self.levels)]
                self.families = {k: [i for i in v if 0 <= i < len(self.levels)]
                                 for k, v in mf.get('families', {}).items()}
                logger.info(f"mouth manifest 载入: openness={len(self.openness)}档 "
                            f"families={ {k: len(v) for k, v in self.families.items()} }")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"manifest.json 解析失败，回退单家族: {e}")
        if not self.openness:
            self.openness = list(range(len(self.levels)))

        # TODO: 至少 2 档的断言（单档无法表现张合）
        if len(self.openness) < 2:
            logger.warning(
                f"mouth_frames 仅 {len(self.openness)} 档，口型张合会失真；"
                f"建议补到 4-8 档: {self.frames_dir}")
        logger.info(f"MouthFrameLibrary 载入 {len(self.levels)} 帧 ← {self.frames_dir}")
        return self

    def get(self, level: int) -> NDArray[np.uint8]:
        """按 openness 序号取嘴帧，越界钳制到 [0, N-1]（旧契约）。"""
        if not self.levels:
            raise RuntimeError("MouthFrameLibrary 未载入任何嘴帧")
        level = min(max(level, 0), len(self.levels) - 1)
        return self.levels[level]

    def get_openness(self, idx: int) -> NDArray[np.uint8]:
        """按张开度序列取帧（manifest 存在时 openness 可能只是全帧子集）。"""
        idx = min(max(idx, 0), len(self.openness) - 1)
        return self.levels[self.openness[idx]]

    def family_pair(self, family: str, f_level: float) -> tuple[NDArray[np.uint8], NDArray[np.uint8], float]:
        """家族内按张开度取相邻两帧 (lo, hi, w)；家族缺帧时逐级回退：
        family→open→openness 全序列。单帧家族（如 closed）w 恒 0。"""
        seq = self.families.get(family)
        if not seq:
            seq = self.families.get('open') or self.openness
        if len(seq) == 1:
            fr = self.levels[seq[0]]
            return fr, fr, 0.0
        # f_level ∈ [0,1] 映射到家族序列内相邻两档
        pos = min(max(f_level, 0.0), 1.0) * (len(seq) - 1)
        lo = min(int(np.floor(pos)), len(seq) - 1)
        hi = min(lo + 1, len(seq) - 1)
        return self.levels[seq[lo]], self.levels[seq[hi]], float(pos - lo)

    @property
    def n_openness(self) -> int:
        return len(self.openness)

    def __len__(self) -> int:
        return len(self.levels)


class RmsToLevelMapper:
    """RMS → 嘴帧档位映射器。

    输入单帧音频能量（RMS），输出嘴帧档位索引；含 attack/release 平滑，
    避免相邻档位间逐帧跳变导致的嘴部抖动。
    """

    def __init__(self, n_levels: int, rms_floor: float = 1e-3, rms_ceiling: float = 0.2,
                 attack: float = 0.35, release: float = 0.12):
        self.n_levels = max(int(n_levels), 1)
        self.rms_floor = float(rms_floor)
        self.rms_ceiling = float(rms_ceiling)
        self.attack = float(attack)      # 升中：张嘴跟随（0.6→0.35 更平滑）
        self.release = float(release)    # 降慢：闭嘴缓释（0.2→0.12 拖尾长一点）
        self._smoothed = 0.0             # 平滑后的归一化能量 ∈ [0,1]

    @staticmethod
    def frame_rms(chunk0: NDArray[np.float32], chunk1: NDArray[np.float32]) -> float:
        """单帧 RMS：拼接 2 个 20ms chunk（合计 40ms = 1 视频帧）。"""
        x = np.concatenate([chunk0, chunk1])
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def __call__(self, rms: float) -> int:
        """归一化 → 平滑 → 量化为档位索引（兼容旧调用）。"""
        lo, hi, w = self.get_weighted(rms)
        # 四舍五入到临近档
        return lo if w < 0.5 else hi

    def get_weighted(self, rms: float) -> tuple[int, int, float]:
        """返回相邻两档及插值权重 (lo, hi, w)，供上层做线性混合。

        相比纯整数档位，可将 10 档查表的嘴形升级为亚像素连续变化。
        """
        n = (rms - self.rms_floor) / max(self.rms_ceiling - self.rms_floor, 1e-6)
        n = min(max(n, 0.0), 1.0)
        alpha = self.attack if n >= self._smoothed else self.release
        self._smoothed += alpha * (n - self._smoothed)
        # 浮点档位
        f_level = self._smoothed * (self.n_levels - 1)
        lo = int(np.floor(f_level))
        hi = lo + 1
        w = float(f_level - lo)
        lo = min(max(lo, 0), self.n_levels - 1)
        hi = min(max(hi, 0), self.n_levels - 1)
        return lo, hi, w

    def reset(self):
        """静音→说话切换时清空平滑状态，避免上句残留影响本句起始口型。"""
        self._smoothed = 0.0


def load_model(path=None):
    """无神经网络：返回 None，保持与 app.py 的 load_model 调用签名兼容。"""
    return None


def load_avatar(avatar_id):
    """复用 wav2lip 的 full_imgs + coords.pkl 作身体与嘴部 bbox；额外加载 mouth_frames。

    不读 face_imgs（无需模型输入）。
    """
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    mouth_frames_path = f"{avatar_path}/mouth_frames"

    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)

    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)

    mouth_lib = MouthFrameLibrary(mouth_frames_path).load()

    return frame_list_cycle, coord_list_cycle, mouth_lib


def warm_up(batch_size, *args, **kwargs):
    """无模型预热：no-op（嘴帧已在 load_avatar 阶段读入内存）。"""
    return


@register("avatar", "mouthdict")
class MouthDictReal(BaseAvatar):
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        self.model = None                                 # 占位，保持 create() 传参一致
        self.frame_list_cycle, self.coord_list_cycle, self.mouth_lib = avatar

        # 允许 --mouth_frames 覆盖默认目录（默认 data/avatars/<id>/mouth_frames）
        _mf = getattr(opt, 'mouth_frames', '') or ''
        if _mf:
            self.mouth_lib = MouthFrameLibrary(_mf).load()

        # 映射器参数取自 opt（config.py 中新增），缺省时回退到内置默认
        # n_levels 用 openness 序列长度（manifest 存在时不含 viseme 家族帧）
        self.mapper = RmsToLevelMapper(
            n_levels=self.mouth_lib.n_openness,
            rms_floor=getattr(opt, 'rms_floor', 0.001),
            rms_ceiling=getattr(opt, 'rms_ceiling', 0.2),
            attack=getattr(opt, 'mouth_attack', 0.6),
            release=getattr(opt, 'mouth_release', 0.2),
        )

        # 仍复用 mel 管线驱动 audio_frames 节拍（零改主循环）
        self.asr = MelASR(opt, self)
        self.asr.warm_up()

        # ---------------- 四层协同子系统 (P0) ----------------
        # 默认开关均为 0 → compositor 仅启用 "mouth" 层（等价旧版 paste_back_frame）
        self.enable_blink = bool(getattr(opt, 'enable_blink', 0))
        self.enable_gaze = bool(getattr(opt, 'enable_gaze', 0))
        self.enable_breath = bool(getattr(opt, 'enable_breath', 0))
        self.enable_emotion = bool(getattr(opt, 'enable_emotion', 0))
        self.enable_occlusion = bool(getattr(opt, 'enable_occlusion', 0))

        self.blink = BlinkScheduler()
        self.gaze = GazeScheduler()
        self.breath = BreathScheduler()
        self.emotion = EmotionMood()
        self.viseme_timeline = VisemeTimeline()
        self.occlusion = OcclusionDetector()

        # 眼帧 / 情绪图集 / timeline 目录约定（P0 不预加载；P1 按存在性懒加载）
        self._avatar_id = getattr(opt, 'avatar_id', '')
        self._eyes_dir = getattr(opt, 'eye_frames_dir', '') or (
            f"./data/avatars/{self._avatar_id}/eyes" if self._avatar_id else '')
        self._expr_dir = getattr(opt, 'expression_dir', '') or (
            f"./data/avatars/{self._avatar_id}/expressions" if self._avatar_id else '')
        self._timeline_dir = getattr(opt, 'viseme_timeline_dir', '') or (
            f"./data/avatars/{self._avatar_id}/viseme_timelines" if self._avatar_id else '')
        self._rhubarb_bin = getattr(opt, 'rhubarb_bin', '') or ''

        # 组装合成器
        self.compositor = LayerCompositor()
        self.compositor.register("breath", self.breath.apply, enabled=self.enable_breath)
        self.compositor.register("mouth", self._paste_mouth_layer, enabled=True)  # 核心层，常开
        self.compositor.register(
            "eyes", self._paste_eyes_layer,
            enabled=(self.enable_blink or self.enable_gaze),
        )
        self.compositor.register("emotion", self.emotion.apply, enabled=self.enable_emotion)

        # P1 预接：懒加载眼帧/情绪图集/timeline（目录不存在时→空，与 P0 pass-through 一致）
        self.eye_frames = self._load_eye_frames(self._eyes_dir)
        self.expression_frames = self._load_expression_frames(self._expr_dir)
        # coords.pkl 可选 eyes_bbox 字段（旧版 pkl 没这个字段→None，不叠眼神）
        self.eyes_bbox_list = self._extract_eyes_bbox(self.coord_list_cycle)
        # 归一化：若 dict 格式，提取 mouth tuple 供 paste_back_frame 使用
        if self.coord_list_cycle and isinstance(self.coord_list_cycle[0], dict):
            self.coord_list_cycle = [c["mouth"] for c in self.coord_list_cycle]
        # 鼠标目标（web 端 POST /mouse 写入）—— None 时 GazeScheduler 回退画面中心
        self.mouse_target = None
        # P1 文本 viseme：utterance 锚点（本句首帧的全局帧号）与待消费 start 事件
        self._viseme_text = bool(getattr(opt, 'enable_viseme_text', 1))
        self._utt_start_frame: int | None = None
        self._utt_total_frames: int = 0
        self._pending_start: tuple[str, int] | None = None   # (text, duration_ms)
        # 句尾检测（mapper._smoothed 从 >0.3 落到 <0.1 的边沿）
        self._last_smoothed_for_edge = 0.0

        # ---------- 交互状态机: speaking / listening / idle ----------
        self._interaction_state = "idle"  # 默认 idle
        # 聆听态参数
        self._listening_blink_scale = 1.8  # 眨眼间隔放大（专注少眨）
        self._listening_breath_scale = 0.5  # 呼吸幅度减半（静立聆听）
        self._listening_nod_period = 4.0    # 每 4s 轻点头一次
        self._last_nod_time = 0.0
        # 加载 breath_idle 素材（nod/lean/tilt 帧，可选）
        self._idle_variant_dir = (
            f"./data/avatars/{self._avatar_id}/breath_idle" if self._avatar_id else '')
        self._idle_variants = self._load_idle_variants(self._idle_variant_dir)

        # 逐帧头部对齐偏移（tools/make_mouth_frames.py 离线产出）：
        # 身体循环头部平移 dx/dy，嘴帧按偏移平移贴回，消除错位双嘴
        self._mouth_align = None
        if self._avatar_id:
            align_path = f"./data/avatars/{self._avatar_id}/mouth_align.pkl"
            if os.path.isfile(align_path):
                with open(align_path, 'rb') as f:
                    self._mouth_align = pickle.load(f)
                logger.info(f"mouth_align 载入 {len(self._mouth_align['offsets'])} 帧偏移")
        # 只贴嘴帧第 70 行以下（鼻/嘴/下巴/脖颈）；眼区保留 body 原图，
        # 避免基准帧静态眼睛覆盖身体帧自然眨眼
        self._mouth_paste_top = 70

        # 内容遮罩：嘴帧相对基准帧（= levels[0]）的差异区才贴。嘴帧底部含
        # 基准帧衣领像素，头部浮动 dy(-22~3) 时整块贴回会把假衣领压到真衣领
        # 上产生错位虚影；差值区天然截止在下唇，衣领/胡须点等相同像素 alpha=0
        self._mouth_mask_f = None
        if self._mouth_align and len(self.mouth_lib) >= 2:
            base0 = self.mouth_lib.levels[0].astype(np.int16)
            union = np.zeros(base0.shape[:2], np.float32)
            for lv in self.mouth_lib.levels:
                d = np.abs(lv.astype(np.int16) - base0).max(axis=2)
                union = np.maximum(union, (d > 12).astype(np.float32))
            union = cv2.dilate(union, np.ones((5, 5), np.uint8))
            union = cv2.GaussianBlur(union, (15, 15), 0)
            self._mouth_mask_f = union[..., None]
            logger.info(f"mouth 内容遮罩生成: 有效像素 {int((union > 0.5).sum())}"
                        f" / {union.size} (差异区=绘制范围+5px 羽化)")

        logger.info(
            f"MouthDictReal layers: active={self.compositor.active_layers()} "
            f"blink={self.enable_blink} gaze={self.enable_gaze} "
            f"breath={self.enable_breath} emotion={self.enable_emotion} "
            f"occlusion={self.enable_occlusion} | "
            f"eye_frames={len(self.eye_frames)} expression_frames={len(self.expression_frames)}"
        )

    # ---------------- P1 预接辅助 ----------------
    @staticmethod
    def _load_eye_frames(eyes_dir: str) -> dict:
        """目录约定：open/closed/closing/opening + 4 方向（left/right/up/down/center）。

        不存在 → 返回空 dict，眼神层 pass-through。
        """
        out: dict = {}
        if not eyes_dir or not os.path.isdir(eyes_dir):
            return out
        keys = ("open", "closed", "closing", "opening",
                "left", "right", "up", "down", "center")
        for k in keys:
            for ext in ("png", "jpg", "jpeg"):
                p = os.path.join(eyes_dir, f"{k}.{ext}")
                if os.path.isfile(p):
                    img = cv2.imread(p)
                    if img is not None:
                        out[k] = img
                    break
        return out

    @staticmethod
    def _load_expression_frames(expr_dir: str) -> dict:
        """目录约定：neutral/happy/surprised/thinking.<ext>，与 EmotionMood.VALID_MOODS 对齐。"""
        out: dict = {}
        if not expr_dir or not os.path.isdir(expr_dir):
            return out
        for k in ("neutral", "happy", "surprised", "thinking", "listening"):
            for ext in ("png", "jpg", "jpeg"):
                p = os.path.join(expr_dir, f"{k}.{ext}")
                if os.path.isfile(p):
                    img = cv2.imread(p)
                    if img is not None:
                        out[k] = img
                    break
        return out

    @staticmethod
    def _extract_eyes_bbox(coord_list) -> list | None:
        """coord_list 支持两种旧格式：
          1. list[(y1,y2,x1,x2)]      → 旧版仅 mouth bbox，无 eyes → 返回 None
          2. list[dict{'mouth':..,'eyes':..}] → P1 新格式，提取 eyes 序列
        """
        if not coord_list:
            return None
        first = coord_list[0]
        if isinstance(first, dict) and "eyes" in first:
            return [c.get("eyes") for c in coord_list]
        return None

    def _rms_from_mel(self, mel_chunk) -> float:
        """无 audio_frames 时的兜底：用 mel 能量作响度代理。"""
        # TODO: mel 为对数谱，非真实 RMS；仅在主循环未传 audio_frames 时降级使用
        return float(np.mean(np.asarray(mel_chunk)))

    # 【接入点 A：跳过 wav2lip 推理】覆盖基类 inference_batch
    def inference_batch(self, index, audiofeat_batch, audio_frames=None):
        """返回 [batch_size, 256, 256, 3] 嘴部贴片，替代神经网络输出。
    
        输出契约与 wav2lip 一致（float 数组，[B,H,W,3]），使基类 paste_back_frame
        的 bbox resize+贴回逻辑可原样复用。
    
        P1 文本 viseme：TTS 在每句首块 eventpoint 带 {status:'start', text, duration_ms}
        → 随 AudioFrameData.userdata 流到这里，命中时建家族时间线并以本帧为锚点；
        形状（家族）由文本定，大小（张开度）由 RMS 定；无时间线/未命中回退纯 RMS。
    
        聆听态：强制 level=0（闭合嘴），偶尔闪 level=1（"嗯"附和）。
        """
        preds = []
        listening = (self._interaction_state == 'listening')
        frame_ms = int(round(1000.0 / max(self.opt.fps, 1)))
        for i in range(self.batch_size):
            if listening:
                # 聆听态：嘴固定闭合，每 ~5s 闪一帧微张（"嗯~"附和）
                t_now = time.monotonic()
                if t_now - self._last_nod_time > self._listening_nod_period:
                    self._last_nod_time = t_now
                    level = 1  # 微张一帧（附和）
                else:
                    level = 0  # 闭合
                preds.append(self.mouth_lib.get_openness(level).astype(np.float32))
                continue
    
            # 贴回用哪张身体帧(idx)由基类 inference 循环决定，这里只产出嘴部贴片
            if audio_frames is not None and i * 2 + 1 < len(audio_frames):
                rms = self.mapper.frame_rms(audio_frames[i * 2].data, audio_frames[i * 2 + 1].data)
            else:
                rms = self._rms_from_mel(audiofeat_batch[i])

            # P1：扫描本 batch 的 utterance start 事件（userdata 随音频帧同步流转）
            if self._viseme_text and audio_frames is not None:
                for j in (i * 2, i * 2 + 1):
                    if j < len(audio_frames):
                        ud = audio_frames[j].userdata or {}
                        if ud.get('status') == 'start' and ud.get('text'):
                            self._pending_start = (ud['text'], int(ud.get('duration_ms', 0) or 0))

            # 新句开始：以当前全局帧为锚点建时间线（同 batch 内后续帧共享锚点）
            if self._pending_start is not None:
                text, dur_ms = self._pending_start
                self._pending_start = None
                if self._viseme_text and dur_ms > 0 and self.viseme_timeline.set_from_text(text, dur_ms):
                    self._utt_start_frame = index + i
                    self._utt_total_frames = max(int(dur_ms / frame_ms), 1)
                    logger.info(f"viseme timeline: '{text[:18]}...' {dur_ms}ms")
                else:
                    self._utt_start_frame = None

            lo, hi, w = self.mapper.get_weighted(rms)
            family = None
            if (self.viseme_timeline.loaded and self._utt_start_frame is not None
                    and (index + i) - self._utt_start_frame <= self._utt_total_frames + 5):
                t_ms = int((index + i - self._utt_start_frame) * frame_ms)
                family = self.viseme_timeline.lookup_family(t_ms)

            if family == 'closed':
                # 闭唇段：张开度直接归零（不受 RMS 拖尾影响）
                pred = self.mouth_lib.family_pair('closed', 0.0)[0].astype(np.float32)
            elif family is not None:
                # 形状由文本定，大小由 RMS 包络定
                f_level = self.mapper._smoothed
                a, b, wgt = self.mouth_lib.family_pair(family, f_level)
                pred = a * (1.0 - wgt) + b * wgt if wgt > 0.001 else a.astype(np.float32)
            else:
                # 回退 RMS 路线：相邻两档线性插值→亚像素连续张合
                a = self.mouth_lib.get_openness(lo).astype(np.float32)
                if w <= 0.001 or lo == hi:
                    pred = a
                else:
                    b = self.mouth_lib.get_openness(hi).astype(np.float32)
                    pred = a * (1.0 - w) + b * w
    
            # 句尾边沿检测：>0.3 → <0.1 时排入一次眨眼（Pl启用 blink 时生效）
            cur = self.mapper._smoothed
            if self.enable_blink and self._last_smoothed_for_edge > 0.3 and cur < 0.1:
                self.blink.force_blink_soon(delay_s=0.02)
            self._last_smoothed_for_edge = cur
    
            preds.append(pred)
        return np.asarray(preds)

    # 【接入点 B：四层合成管线】覆盖基类；当所有新开关=0 时与旧版逐帧等价
    def paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]
        # 防御：dict 格式 coords.pkl（即使 __init__ 归一化未执行仍可兼容）
        if isinstance(bbox, dict):
            bbox = bbox["mouth"]
        body = self.frame_list_cycle[idx].copy()
        ctx = {
            "idx": idx,
            "mouth_bbox": bbox,
            "pred_frame": pred_frame,
            "now": time.monotonic(),
            "smoothed_rms": float(self.mapper._smoothed),
            "frame_size": (body.shape[1], body.shape[0]),
            # P1 预接：有素材时自动向下传递，无素材时 scheduler pass-through
            "eye_bbox": (self.eyes_bbox_list[idx] if self.eyes_bbox_list else None),
            "eye_frames": self.eye_frames or None,
            "expression_frames": self.expression_frames or None,
            "mouse": self.mouse_target,
        }
        return self.compositor.compose(body, ctx)

    # ---------------- 外部控制 API ----------------
    def set_mouse_target(self, x: float, y: float) -> None:
        """web 端 POST /mouse 调用。无 session 时 GazeScheduler 回退画面中心。"""
        try:
            self.mouse_target = (float(x), float(y))
        except (TypeError, ValueError):
            self.mouse_target = None

    def set_mood(self, name: str, fade_ms: int | None = None) -> None:
        """web 端 POST /mood 或 P2 LLM 意图总线调用。"""
        if fade_ms is None:
            self.emotion.set_mood(name)
        else:
            self.emotion.fade_to(name, ms=fade_ms)

    def set_interaction_state(self, state: str) -> None:
        """外部控制交互态: 'speaking'|'listening'|'idle'。由 /set_state 端点调用。"""
        valid = ('speaking', 'listening', 'idle')
        if state not in valid:
            logger.warning(f"set_interaction_state: invalid '{state}', expect {valid}")
            return
        old = self._interaction_state
        self._interaction_state = state
        if state == 'listening':
            # 进入聆听：切换到 listening 表情（若有素材）
            if self.enable_emotion and 'listening' in self.expression_frames:
                self.emotion.set_mood('listening')
            # 眨眼间隔放大
            self.blink.blink_min_s *= self._listening_blink_scale
            self.blink.blink_max_s *= self._listening_blink_scale
            # 呼吸幅度减半
            self.breath.amp *= self._listening_breath_scale
        elif old == 'listening':
            # 离开聆听：恢复参数
            self.blink.blink_min_s /= self._listening_blink_scale
            self.blink.blink_max_s /= self._listening_blink_scale
            self.breath.amp /= self._listening_breath_scale
            # 恢复默认表情
            if self.enable_emotion:
                self.emotion.set_mood('neutral')
        logger.info(f"interaction_state: {old} → {state}")

    @staticmethod
    def _load_idle_variants(idle_dir: str) -> dict:
        """加载 breath_idle/ 目录下的 nod.png, lean.png, tilt.png。"""
        out = {}
        if not idle_dir or not os.path.isdir(idle_dir):
            return out
        for name in ('nod', 'lean', 'tilt'):
            for ext in ('.png', '.jpg'):
                p = os.path.join(idle_dir, name + ext)
                if os.path.isfile(p):
                    img = cv2.imread(p)
                    if img is not None:
                        out[name] = img
                    break
        return out

    # ------ 内部层回调（供 LayerCompositor 使用）------
    def _paste_mouth_layer(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """L1 嘴形层：逐帧头部位移对齐 + 只贴眼以下区域 + 羽化边缘。

        无 mouth_align.pkl 时回退旧版整块硬贴（保持向后兼容）。
        开启遮挡时命中→跳过贴图。
        """
        if self.enable_occlusion:
            bbox = ctx["mouth_bbox"]
            if self.occlusion.detect(frame, bbox):
                # 遮挡帧：不贴嘴，保留 body
                return frame
        bbox = ctx["mouth_bbox"]
        pred = ctx["pred_frame"]
        y1, y2, x1, x2 = bbox
        h, w = y2 - y1, x2 - x1
        res_full = cv2.resize(pred.astype(np.uint8), (w, h))

        if not self._mouth_align:
            frame[y1:y2, x1:x2] = res_full
            return frame

        offs = self._mouth_align["offsets"]
        dx, dy = offs[ctx["idx"]] if ctx["idx"] < len(offs) else (0, 0)
        top = min(self._mouth_paste_top, h - 1)
        res = res_full[top:, :]                      # 鼻/嘴/下巴/脖颈
        ph, pw = res.shape[:2]

        cache = getattr(self, "_feather_cache", None)
        if cache is None or cache[0] != (ph, pw):
            f = np.ones((ph, pw), dtype=np.float32)
            fe = 10
            for i in range(fe):
                v = i / fe
                f[i, :] = np.minimum(f[i, :], v)
                f[ph - 1 - i, :] = np.minimum(f[ph - 1 - i, :], v)
                f[:, i] = np.minimum(f[:, i], v)
                f[:, pw - 1 - i] = np.minimum(f[:, pw - 1 - i], v)
            if self._mouth_mask_f is not None:
                m = self._mouth_mask_f
                if m.shape[:2] != (h, w):   # 嘴帧尺寸与 bbox 不一致时兜底缩放
                    m = cv2.resize(m, (w, h))
                f = f * m[top:top + ph, :, 0]
            self._feather_cache = ((ph, pw), f[..., None])
            cache = self._feather_cache
        alpha = cache[1]

        # 目标矩形（平移后钳制到画面内）
        H, W = frame.shape[:2]
        px1, py1 = x1 + dx, y1 + top + dy
        dx1, dy1 = max(px1, 0), max(py1, 0)
        dx2, dy2 = min(px1 + pw, W), min(py1 + ph, H)
        if dx2 <= dx1 or dy2 <= dy1:
            return frame
        sx1, sy1 = dx1 - px1, dy1 - py1
        sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
        roi = frame[dy1:dy2, dx1:dx2].astype(np.float32)
        a = alpha[sy1:sy2, sx1:sx2]
        frame[dy1:dy2, dx1:dx2] = (
            res[sy1:sy2, sx1:sx2].astype(np.float32) * a + roi * (1.0 - a)
        ).astype(np.uint8)
        return frame
    
    def _paste_eyes_layer(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """L2 眼神层：串联 BlinkScheduler + GazeScheduler。
    
        P0/P1：eye_bbox / eye_frames 任一缺失 → 只推进状态机、返回入参帧。
        眨眼帧优先于视线偏移（眨眼期间不叠视线，避免贴图互盖）。
        """
        # 向 ctx 补充 eye_frames（blink.apply 从 ctx 里取）
        if "eye_frames" not in ctx:
            ctx["eye_frames"] = self.eye_frames or None
        # blink.apply 内部会选 eye_frames["closed"] 等；gaze 需要 gaze_frames
        # 为简化 P1，将 gaze 四方向帧从 eye_frames 同一字典里取（目录约定兼容）
        if "gaze_frames" not in ctx:
            ctx["gaze_frames"] = self.eye_frames or None
        out = self.blink.apply(frame, ctx)
        eye_state = ctx.get("eye_state", "open")
        if eye_state != "open":
            # 非瞫眼状态不再叠视线偏移（防止睁-闭帧与左右眼帧互相覆盖）
            self.gaze.update(ctx.get("now"))
            return out
        return self.gaze.apply(out, ctx)
