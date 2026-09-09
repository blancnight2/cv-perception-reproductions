# -*- coding: utf-8 -*-
"""5 张图 grounding 可行性验证：验显存 / 验 JSON / 验坐标口径"""
import torch, json, re, glob, os
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

VAL_IMG = '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/images/val'
OUT_DIR = os.path.expanduser('~/openvocab/check')
MODEL   = 'Qwen/Qwen3-VL-8B-Instruct'
os.makedirs(OUT_DIR, exist_ok=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
proc  = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
            MODEL, quantization_config=bnb, device_map='auto').eval()
print('显存占用 %.1f GB' % (model.get_memory_footprint() / 1e9))

PROMPT = ('Detect every car, pedestrian and cyclist in this image.\n'
          'Output ONLY a JSON array. No explanation, no markdown fence.\n'
          'Each element must be {"bbox_2d":[x1,y1,x2,y2],"label":"car|person|bicycle"}')

FENCE = re.compile(r'^```(?:json)?|```$', re.M)

def parse_boxes(text):
    """从模型输出里稳健地抠出 JSON 数组"""
    t = FENCE.sub('', text.strip()).strip()
    m = re.search(r'\[.*\]', t, flags=re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for e in arr:
        if not isinstance(e, dict):
            continue
        b = e.get('bbox_2d') or e.get('bbox')
        if isinstance(b, list) and len(b) == 4:
            try:
                out.append(([float(v) for v in b], str(e.get('label', '')).lower()))
            except (TypeError, ValueError):
                continue
    return out


imgs = sorted(glob.glob(os.path.join(VAL_IMG, '*')))[:5]
print('找到 %d 张图' % len(imgs))
n_ok, stats = 0, []

for path in imgs:
    img = Image.open(path).convert('RGB')
    W, H = img.size
    msgs = [{'role': 'user', 'content': [
                {'type': 'image', 'image': path},
                {'type': 'text',  'text': PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    ans = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True)[0]

    boxes = parse_boxes(ans)
    print('\n--- %s  (%dx%d) ---' % (os.path.basename(path), W, H))
    print('原始输出:', ans[:200].replace('\n', ' '))
    print('解析出 %d 个框' % len(boxes))
    if not boxes:
        continue
    n_ok += 1
    mx = max(max(b[0]) for b in boxes)
    stats.append((mx, W, H))
    print('坐标最大值 %.1f   图宽 %d   图高 %d' % (mx, W, H))

    dr = ImageDraw.Draw(img)
    for b, lab in boxes:
        dr.rectangle(b, outline=(255, 0, 0), width=3)
        dr.text((b[0], max(0, b[1] - 12)), lab, fill=(255, 0, 0))
    img.save(os.path.join(OUT_DIR, os.path.basename(path)))

print('\n' + '=' * 52)
print('5 张里成功解析出框的: %d/5' % n_ok)
if stats:
    mx, W, H = stats[0]
    if mx <= 1000 and W > 1100:
        print('>> 坐标疑似【0-1000 归一化】 → 全量评测时 NORMALIZED = True')
    else:
        print('>> 坐标疑似【绝对像素】 → 全量评测时 NORMALIZED = False')
print('>> 画好框的图在 %s ，务必人工打开看一眼' % OUT_DIR)
print('=' * 52)
