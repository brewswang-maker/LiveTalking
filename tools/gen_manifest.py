#!/usr/bin/env python3
"""Generate asset_manifest.md for huadun avatar."""
import os, datetime

base = 'data/avatars/huadun'
lines = []
lines.append('# Asset Manifest \u2014 huadun\n')
lines.append(f'\u751f\u6210\u65f6\u95f4: {datetime.datetime.now().isoformat()}\n')
lines.append('\u6e90\u7d20\u6750: /Users/mac1234/\u9879\u76ee/\u6570\u5b57\u4eba/\u53ea\u6709logo/ (8 \u6bb5\u89c6\u9891)\n')

sections = [
    ('mouth_frames', '\u53e3\u578b viseme \u8868', '256\u00d7256 \u5634\u533a crop (bbox 300,600\u2192460,670)\uff0c\u6309\u5f20\u5f00\u5ea6\u5347\u5e8f 0-9'),
    ('eyes', '\u773c\u90e8\u56fe\u96c6', '256\u00d7256 \u773c\u533a crop (bbox 265,465\u2192495,555)\uff0copen/closed/\u65b9\u5411'),
    ('expressions', '\u60c5\u7eea\u56fe\u96c6', '720\u00d71280 \u5168\u8eab\u5e27\uff0c\u6309\u60c5\u7eea\u547d\u540d'),
    ('breath_idle', '\u547c\u5438/\u70b9\u5934 idle', '720\u00d71280 \u5168\u8eab\u5e27\uff0cnod/lean/tilt'),
    ('full_imgs', '\u8bf4\u8bdd\u4e3b\u5faa\u73af\u5e27', '720\u00d71280\uff0c\u6e90: \u5634\u578b2.mp4 (251\u5e27, 10s)'),
    ('face_imgs', '\u5bf9\u5e94\u5634\u533a\u88c1\u526a', '256\u00d7256\uff0c\u4e0e full_imgs \u5e27\u53f7\u4e00\u4e00\u5bf9\u5e94'),
]
for dirname, title, desc in sections:
    d = os.path.join(base, dirname)
    if os.path.isdir(d):
        files = sorted(os.listdir(d))
        lines.append(f'\n## {title} (`{dirname}/`)\n')
        lines.append(f'\u89c4\u683c: {desc}\n')
        lines.append(f'\u6570\u91cf: {len(files)} \u6587\u4ef6\n')
        for f in files[:15]:
            sz = os.path.getsize(os.path.join(d, f))
            lines.append(f'- `{f}` ({sz//1024}KB)\n')
        if len(files) > 15:
            lines.append(f'- ... \u5171 {len(files)} \u4e2a\n')

lines.append('\n## coords.pkl\n')
lines.append("- \u683c\u5f0f: `list[dict]`\uff0c\u6bcf\u9879 `{'mouth': (y1,y2,x1,x2), 'eyes': (y1,y2,x1,x2)}`\n")
lines.append('- 251 entries\uff08\u4e0e full_imgs \u5e27\u6570\u4e00\u81f4\uff09\n')
lines.append('- mouth bbox: (y1=600, y2=670, x1=300, x2=460)\n')
lines.append('- eyes bbox: (y1=465, y2=555, x1=265, x2=495)\n')
lines.append('- \u4ee3\u7801\u517c\u5bb9: MouthDictReal.__init__ \u81ea\u52a8\u5f52\u4e00\u5316 dict\u2192tuple\n')

lines.append('\n## \u52a8\u4f5c\u6ce8\u518c (custom.json)\n')
lines.append('- audiotype=1: `data/avatars/huadun_idle/` (101\u5e27, \u5f85\u673a.mp4)\n')
lines.append('- audiotype=2: `data/avatars/huadun_salute/` (101\u5e27, \u656c\u793c.mp4) + \u7b49\u957f\u9759\u97f3 wav\n')

lines.append('\n## \u89c6\u9891\u6e90\u5bf9\u5e94\u5173\u7cfb\n')
lines.append('| \u89c6\u9891 | \u65f6\u957f | \u5e27\u6570(25fps) | \u7528\u9014 |\n')
lines.append('|------|------|------------|------|\n')
lines.append('| \u5634\u578b2.mp4 | 10s | 251 | mouth_frames \u4e3b\u6e90 + full_imgs |\n')
lines.append('| \u53e3\u578b.mp4 | 10s | 251 | eyes open/closed \u6e90 |\n')
lines.append('| \u601d\u8003\u3001\u6216\u542c.mp4 | 5s | 126 | eyes \u65b9\u5411\u5e27 + thinking + breath_idle |\n')
lines.append('| \u542c.mp4 | 5s | 126 | listening + breath_idle/tilt |\n')
lines.append('| \u65e0\u6cd5\u56de\u7b54.mp4 | 4s | 101 | surprised |\n')
lines.append('| \u8bb2\u89e3.mp4 | 4s | 101 | happy |\n')
lines.append('| \u656c\u793c.mp4 | 4s | 101 | audiotype=2 \u52a8\u4f5c |\n')
lines.append('| \u5f85\u673a.mp4 | 4s | 101 | audiotype=1 idle + neutral |\n')

lines.append('\n## \u7f3a\u53e3/\u9650\u5236\n')
lines.append('### \u5df2\u89e3\u51b3\n')
lines.append('- \u2705 \u53e3\u578b 10 \u6863\uff08\u95ed\u5408\u2192\u5927\u5f20\u68af\u5ea6\u6e05\u6670\uff09\n')
lines.append('- \u2705 \u773c\u5e27 open/closed + 4 \u65b9\u5411\uff087 \u6587\u4ef6\uff09\n')
lines.append('- \u2705 \u60c5\u7eea 5 \u5957\uff08neutral/happy/surprised/thinking/listening\uff09\n')
lines.append('- \u2705 \u8046\u542c\u6001\u7d20\u6750\uff08\u542c.mp4 + \u601d\u8003\u6216\u542c.mp4\uff09\n')
lines.append('- \u2705 \u656c\u793c/\u5f85\u673a\u52a8\u4f5c\u6ce8\u518c\n')
lines.append('- \u2705 coords.pkl \u542b eyes_bbox\uff08\u65b0 dict \u683c\u5f0f\uff09\n')

lines.append('\n### \u4ecd\u5b58\u9650\u5236\uff08\u975e\u963b\u585e\uff09\n')
lines.append('- \u26a0\ufe0f **O/U \u5706\u5507 viseme**: \u7d20\u6750\u5634\u5f62\u4ee5\u201c\u5fae\u7b11\u5f00\u5408\u201d\u4e3a\u4e3b\uff0c\u7f3a\u5c11\u660e\u786e\u5706\u5507(O/U)\u548c\u6241\u5507(EE)\u3002')
lines.append('  \u5f71\u54cd: \u53d1\u201c\u4e4c/\u54e6/\u4e00\u201d\u65f6\u5634\u5f62\u504f\u5dee\uff0c\u4f46\u89c2\u4f17\u611f\u77e5\u201c\u8282\u594f\u5bf9\u201d\u5373\u53ef\u3002')
lines.append('  \u5efa\u8bae: \u540e\u7eed\u7f8e\u672f\u8865\u62cd 2-3 \u5e27\u9759\u6001\uff1a\u5706\u5507O\u3001\u6241\u5507EE\u3001\u53cc\u5507MBP\u3002\n')
lines.append('- \u26a0\ufe0f **\u773c\u5e27\u65b9\u5411\u5dee\u5f02\u5c0f**: \u5361\u901a\u89d2\u8272\u77b3\u5b54\u4f4d\u79fb\u4ec5 \u00b15px\uff0c\u5b9e\u9645\u89c6\u7ebf\u53d8\u5316\u4e3b\u8981\u4f9d\u8d56 GazeScheduler warpAffine \u504f\u79fb\u3002\n')
lines.append('- \u26a0\ufe0f **\u8868\u60c5\u5e27\u5934\u4f4d\u5fae\u504f**: surprised/thinking \u5e27\u6709 \u00b13\u00b0 \u5934\u4f4d\u504f\u79fb\uff0cEmotionMood addWeighted \u878d\u5408\u53ef\u5bb9\u5fcd\u3002\n')

with open(os.path.join(base, 'asset_manifest.md'), 'w', encoding='utf-8') as f:
    f.write(''.join(lines))
print('\u2713 asset_manifest.md written')
