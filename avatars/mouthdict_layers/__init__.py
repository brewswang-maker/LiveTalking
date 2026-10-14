###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#
#  mouthdict 四层协同子包：LayerCompositor 与 6 个 scheduler。
#
#  设计约定（P0 骨架）：
#  - 每个 scheduler 在"启用开关=关"或"素材缺失"时，必须显式返回入参帧不变
#    （pass-through）；上层 LayerCompositor 依赖此契约保证与旧版逐帧等价。
#  - 单帧合成预算 <10ms；仅用 numpy / cv2 原地或浅拷贝操作，零 GPU 推理。
#  - 本包不 import base_avatar / wav2lip 等其他 avatar 子类，避免跨子类耦合。
###############################################################################

from avatars.mouthdict_layers.compositor import LayerCompositor
from avatars.mouthdict_layers.blink_scheduler import BlinkScheduler
from avatars.mouthdict_layers.gaze_scheduler import GazeScheduler
from avatars.mouthdict_layers.breath_scheduler import BreathScheduler
from avatars.mouthdict_layers.emotion_mood import EmotionMood
from avatars.mouthdict_layers.viseme_timeline import VisemeTimeline
from avatars.mouthdict_layers.occlusion import OcclusionDetector

__all__ = [
    "LayerCompositor",
    "BlinkScheduler",
    "GazeScheduler",
    "BreathScheduler",
    "EmotionMood",
    "VisemeTimeline",
    "OcclusionDetector",
]
