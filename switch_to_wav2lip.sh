#!/usr/bin/env bash
# =============================================================================
# 一键切回 wav2lip + 讲解 body 方案
# 前提：讲解.mp4 存在于 --video 指定的路径
#
# 用法：
#   bash switch_to_wav2lip.sh /绝对路径/讲解.mp4
#
# 会做 4 件事：
#   1) 停当前 app.py 进程
#   2) 用讲解.mp4 重建 data/avatars/huadun/{full_imgs,face_imgs,coords.pkl}
#   3) 清 mouthdict 遗留的 dict 格式 coords.pkl → 写 tuple 格式（wav2lip 友好）
#   4) 用 wav2lip 启动数字人（后台 nohup，日志写 livetalking.log）
# =============================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <讲解.mp4 路径>"
  exit 1
fi
JIESHAO_VIDEO="$1"
[[ -f "$JIESHAO_VIDEO" ]] || { echo "✗ 视频不存在: $JIESHAO_VIDEO"; exit 1; }

cd "$(dirname "$0")"
echo "== [1/4] 停当前 app.py =="
pkill -f 'python.*app\.py' 2>/dev/null && echo "  已停" || echo "  无进程"
sleep 2

echo ""
echo "== [2/4] 重建 huadun body（讲解.mp4 抽帧）=="
python3 tools/rebuild_body_from_jieshao.py --video "$JIESHAO_VIDEO" --avatar-id huadun

echo ""
echo "== [3/4] 确认 coords.pkl 是 tuple 格式 =="
python3 -c "
import pickle
d = pickle.load(open('data/avatars/huadun/coords.pkl','rb'))
print(f'  count={len(d)}, first_type={type(d[0]).__name__}, first={d[0]}')
assert not isinstance(d[0], dict), 'coords.pkl 仍是 dict，重建未生效'
print('  ✓ tuple 格式')
"

echo ""
echo "== [4/4] 启动 wav2lip 数字人 =="
nohup python app.py \
  --transport webrtc \
  --model wav2lip \
  --avatar_id huadun \
  --modelres 256 \
  --batch_size 4 \
  --customvideo_config custom.json \
  --kb_path data/huadun_kb.json \
  > livetalking.log 2>&1 &

APP_PID=$!
echo "  app.py pid=$APP_PID"
sleep 4
if kill -0 "$APP_PID" 2>/dev/null; then
  echo "  ✓ 启动成功，端口 8010"
  echo "  访问 http://<服务器IP>:8010/huadun_reception.html"
  echo "  tail -f livetalking.log 观察日志"
else
  echo "  ✗ 启动失败，最后 20 行日志："
  tail -20 livetalking.log
  exit 1
fi
