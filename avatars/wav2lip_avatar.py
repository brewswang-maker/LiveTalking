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
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch, audio_frames=None):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        # v5 遮罩：只抹嘴部矩形（core y124~200 x48~208，外羽化 8px），
        # 下巴/脖颈/衣领保留原图作强条件 → 网络非嘴区透传拷贝≈输入，
        # 贴回渐变带因此不可见，从根本上消除颈/领残影与逐帧跳变
        m = getattr(self, "_mouth_mask", None)
        if m is None or m.shape[:2] != img_batch.shape[1:3]:
            mh, mw = img_batch.shape[1], img_batch.shape[2]
            cy0, cy1, cx0, cx1 = 124, 200, 48, 208
            yy, xx = np.mgrid[0:mh, 0:mw]
            dy = np.maximum(np.maximum(cy0 - yy, yy - (cy1 - 1)), 0)
            dx = np.maximum(np.maximum(cx0 - xx, xx - (cx1 - 1)), 0)
            m = np.clip(1.0 - np.maximum(dy, dx) / 8.0, 0, 1).astype(np.float32)
            self._mouth_mask = m
        img_masked = (img_batch * (1.0 - m)[None, :, :, None]).astype(img_batch.dtype)

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        """软融合贴回 v6（v5 静态 alpha + pred 偏差逐帧动态保护）：
        - 顶部：y<23.5% 保留眼睛/额头原图，23.5%~37% 过渡到 pred
        - 列窗：pred 限于 x 56~200（嘴角并集），两侧羽化到 0.03
        - 底部：232 行起渐变到 0.03（bbox 底边回原图）
        - v6 动态保护：嘴下区（y≥200）网络透传拷贝失真（截图长方形虚影），
          按 pred 与原图偏差逐像素压制 alpha，高斯+EMA 时间平滑防逐帧跳；
          嘴驱动区（y<200）不保护
        """
        bbox = self.coord_list_cycle[idx]
        # 兼容 dict 格式 coords.pkl（P1 升级后含 mouth/eyes）
        if isinstance(bbox, dict):
            bbox = bbox["mouth"]
        y1, y2, x1, x2 = bbox
        h, w = y2 - y1, x2 - x1
        body = self.frame_list_cycle[idx].copy()
        res = cv2.resize(pred_frame.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)

        # 静态分量缓存：行顶部 profile + 列窗 profile
        cache = getattr(self, "_alpha_cache", None)
        if cache is None or cache[0] != (h, w):
            base_top = np.full((h,), 1.0, dtype=np.float32)
            y_eye_end = int(h * 0.235)
            y_mouth_top = int(h * 0.37)
            base_top[:y_eye_end] = 0.03
            for y in range(y_eye_end, y_mouth_top):
                base_top[y] = 0.03 + 0.97 * ((y - y_eye_end) / max(y_mouth_top - y_eye_end - 1, 1))
            colprof = np.full((w,), 1.0, dtype=np.float32)
            x_l0, x_l1, x_r0, x_r1 = 36, 56, 200, 220
            colprof[:x_l0] = 0.03
            colprof[x_r1:] = 0.03
            for x in range(x_l0, x_l1):
                colprof[x] = 0.03 + 0.97 * ((x - x_l0) / max(x_l1 - x_l0 - 1, 1))
            for x in range(x_r0, x_r1):
                colprof[x] = 1.0 - 0.97 * ((x - x_r0) / max(x_r1 - x_r0 - 1, 1))
            self._alpha_cache = ((h, w), base_top, colprof)
            cache = self._alpha_cache
        base_top, colprof = cache[1], cache[2]

        roi_u8 = body[y1:y2, x1:x2]
        # 行底部 profile：232 以上 1.0，232~250 渐变到 0.03（bbox 底边回原图）
        rowbot = np.full((h,), 0.03, dtype=np.float32)
        y0 = min(232, h - 1)
        rowbot[:y0] = 1.0
        for y in range(y0, min(250, h)):
            rowbot[y] = 1.0 - 0.97 * ((y - y0) / max(min(250, h) - y0 - 1, 1))
        alpha = ((base_top * rowbot)[:, None] * colprof[None, :])[..., None]

        # v6 动态保护：嘴下区（y≥200）pred 与原图偏差大 = 网络透传失真（长方形虚影），
        # 逐像素压制 alpha；高斯平滑边界 + EMA 时间平滑消除逐帧跳变
        diff = np.abs(res.astype(np.int16) - roi_u8.astype(np.int16)).mean(axis=2)
        prot = np.clip((diff - 8.0) / 24.0, 0, 1).astype(np.float32)
        prot[:min(200, h), :] = 0.0
        prot = cv2.GaussianBlur(prot, (5, 5), 0)
        ema = getattr(self, "_prot_ema", None)
        if ema is None or ema.shape != prot.shape:
            self._prot_ema = prot
        else:
            self._prot_ema = 0.7 * ema + 0.3 * prot
        alpha = alpha * (1.0 - 0.97 * self._prot_ema)[..., None]

        roi = roi_u8.astype(np.float32)
        body[y1:y2, x1:x2] = (res.astype(np.float32) * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        return body

