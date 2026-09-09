# -*- coding: utf-8 -*-
"""诊断版：保存原始输出 + 按归一化换算后画框，定位解析失败原因"""
import torch, json, re, glob, os
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

VAL_IMG = '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/images/val'
OUT_DIR = os.path.expanduser('~/openvocab/diag')
RAW_DIR = os.path.expanduser('~/openvocab/raw')
MODEL   = 'Qwen/Qwen3-VL-8B-Instruct'
NORMALIZED = True          # 上一轮已确认
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
proc  = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
            MODEL, quantization_config=bnb, device_map='auto').eval()

PROMPT = ('Detect every car, pedestrian and cyclist in this image.\n'
          'Output ONLY a JSON array. No explanation, no markdown fence.\n'
          'Each element must be {"bbox_2d":[x1,y1,x2,y2],"label":"car|person|bicycle"}')

FENCE = re.compile(r'^```(?:json)?|```$', re.M)


def parse_boxes(text):
    """返回 (boxes, 失败原因)。boxes 为 [] 且原因非空 = 解析失败"""
    t = FENCE.sub('', text.strip()).strip()
    m = re.search(r'\[.*\]', t, flags=re.S)
    if not m:
        return [], 'no_json_array'
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], 'json_decode_error: %s' % e
    if not isinstance(arr, list):
        return [], 'not_a_list'
    if len(arr) == 0:
        return [], 'empty_array(模型认为图里没有目标)'
    out, bad = [], 0
    for e in arr:
        if not isinstance(e, dict):
            bad += 1; continue
        b = e.get('bbox_2d') or e.get('bbox')
        if isinstance(b, list) and len(b) == 4:
            try:
                out.append(([float(v) for v in b], str(e.get('label', '')).lower()))
            except (TypeError, ValueError):
                bad += 1
        else:
            bad += 1
    if not out:
        return [], 'no_valid_bbox_field (共 %d 条都不合格，样例键=%s)' % (
            len(arr), list(arr[0].keys()) if isinstance(arr[0], dict) else type(arr[0]))
    return out, ''


imgs = sorted(glob.glob(os.path.join(VAL_IMG, '*')))[:5]
summary = []

for path in imgs:
    stem = os.path.splitext(os.path.basename(path))[0]
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

    open(os.path.join(RAW_DIR, stem + '.txt'), 'w').write(ans)

    boxes, reason = parse_boxes(ans)
    print('\n' + '=' * 60)
    print('%s  %dx%d   框数=%d   %s' % (stem, W, H, len(boxes),
                                        ('失败: ' + reason) if reason else 'OK'))
    print('--- 原始输出前 400 字 ---')
    print(ans[:400])
    summary.append((stem, len(boxes), reason))

    if not boxes:
        continue

    dr = ImageDraw.Draw(img)
    for b, lab in boxes:
        if NORMALIZED:
            b = [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]
        dr.rectangle(b, outline=(255, 0, 0), width=3)
        dr.text((b[0], max(0, b[1] - 12)), lab, fill=(255, 255, 0))
    img.save(os.path.join(OUT_DIR, stem + '.png'))

print('\n' + '#' * 60)
for stem, n, reason in summary:
    print('%-10s 框=%-3d %s' % (stem, n, reason or 'OK'))
print('原始输出全文在 %s' % RAW_DIR)
print('换算后画框的图在 %s' % OUT_DIR)
print('#' * 60)
