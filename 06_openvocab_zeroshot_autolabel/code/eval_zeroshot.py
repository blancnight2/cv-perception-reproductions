# -*- coding: utf-8 -*-
"""
KITTI val 全量零样本评测 —— 所有模型走同一套评测代码，保证数字可比。

用法:
    python eval_zeroshot.py --backend qwen  --limit 50      # 先小样本试跑
    python eval_zeroshot.py --backend qwen                  # 全量 1496
    python eval_zeroshot.py --backend yoloe
    python eval_zeroshot.py --backend yoloe --pf            # YOLOE 无提示模式

跑之前必须先跑 test5.py 确认坐标口径，然后把下面的 NORMALIZED 改对。
"""
import argparse, os, glob, json, re, sys
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

# ======================= 配置 =======================
VAL_IMG = os.environ.get('VAL_IMG',
    '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/images/val')
VAL_LBL = os.environ.get('VAL_LBL',
    '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/labels/val')

# kitti.yaml: names: {0: Car, 1: Pedestrian, 2: Cyclist}
NAMES = ['Car', 'Pedestrian', 'Cyclist']

# 文本标签 -> class id。同义词都映射过去，避免模型换个词就丢框
T2ID = {
    'car': 0, 'vehicle': 0, 'automobile': 0, 'sedan': 0, 'truck': 0, 'van': 0,
    'person': 1, 'pedestrian': 1, 'people': 1, 'human': 1, 'man': 1, 'woman': 1,
    'bicycle': 2, 'cyclist': 2, 'bike': 2, 'biker': 2, 'motorcycle': 2,
}

# ⚠️ 跑完 test5.py 后按它的提示改这一项
NORMALIZED = True         # 2026-09-02 实测确认 Qwen3-VL 输出 0-1000 归一化坐标

QWEN_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
YOLOE_MODEL = 'yoloe-11l-seg.pt'
YOLOE_PF_MODEL = 'yoloe-11l-seg-pf.pt'
CONF = 0.25

PROMPT = chr(10).join([
    'Detect objects in this driving scene using EXACTLY these three categories:',
    '- car: any car, van, minivan or small truck',
    '- pedestrian: a person walking or standing, NOT riding a bicycle',
    '- cyclist: a person riding a bicycle. The box MUST cover the person AND',
    '  the bicycle together as ONE object, not two separate boxes.',
    'List AT MOST 25 objects, ordered by confidence (most certain first).',
    'Include small and distant objects too, but do NOT invent objects.',
    'Do NOT output duplicate or overlapping boxes for the same object.',
    'Output ONLY a JSON array. No explanation, no markdown fence.',
    'Each element must be {"bbox_2d":[x1,y1,x2,y2],"label":"car|pedestrian|cyclist"}',
])

FENCE = re.compile(r'^```(?:json)?|```$', re.M)
OBJ   = re.compile(r'\{[^{}]*\}')          # 截断兜底：逐对象抠取


# ======================= 真值读取 =======================
def load_gt(stem, W, H):
    """YOLO txt (cls cx cy w h, 归一化) -> xyxy 像素"""
    f = os.path.join(VAL_LBL, stem + '.txt')
    boxes, labels = [], []
    if os.path.exists(f):
        for line in open(f):
            v = line.split()
            if len(v) < 5:
                continue
            c = int(v[0])
            cx, cy, w, h = (float(x) for x in v[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
            labels.append(c)
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long))


# ======================= 预测器 =======================
class QwenPredictor:
    name = 'Qwen3-VL'

    def __init__(self, model_id=QWEN_MODEL):
        from transformers import AutoProcessor, BitsAndBytesConfig
        import transformers as _tf
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        # Qwen3.5/3.6 原生多模态用 AutoModelForMultimodalLM；
        # Qwen3-VL 用 AutoModelForImageTextToText。按 id 自动选，找不到就回退。
        want_mm = ('Qwen3.6' in model_id) or ('Qwen3.5' in model_id)
        cls = None
        for name in (['AutoModelForMultimodalLM', 'AutoModelForImageTextToText']
                     if want_mm else
                     ['AutoModelForImageTextToText', 'AutoModelForMultimodalLM']):
            cls = getattr(_tf, name, None)
            if cls is not None:
                print('加载类：%s' % name)
                break
        if cls is None:
            raise RuntimeError('transformers 版本过旧，两个多模态 AutoModel 都没有')
        self.image_key = 'url' if want_mm else 'image'
        self.proc = AutoProcessor.from_pretrained(model_id)
        # device_map='auto' 在 Qwen3.6-35B-A3B 上估算错误会甩层到 CPU，
        # 导致 bnb 4-bit 报 "Some modules are dispatched on the CPU or the disk"。
        # 显存已核算够用 -> 全放 0 号卡，绕开自动估算；dtype 必须显式给。
        _kw = dict(quantization_config=bnb, device_map={'': 0},
                   low_cpu_mem_usage=True)
        try:
            self.model = cls.from_pretrained(
                model_id, dtype=torch.bfloat16, **_kw).eval()
        except TypeError:                      # 旧版 transformers 用 torch_dtype
            self.model = cls.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, **_kw).eval()
        self.name = model_id.split('/')[-1]
        self.n_fail = 0
        self.n_salvage = 0
        print('显存占用 %.1f GB' % (self.model.get_memory_footprint() / 1e9))

    def _parse(self, text):
        """两级解析：先整体 JSON，失败则逐对象兜底（救回截断前的完整框）"""
        t = FENCE.sub('', text.strip()).strip()
        arr = None
        m = re.search(r'\[.*\]', t, flags=re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
            except json.JSONDecodeError:
                arr = None
        if arr is None:
            # 输出被 max_new_tokens 截断时，整体数组不合法。
            # 逐个抠出完整的 {...} 片段，能救回截断点之前的全部框。
            arr = []
            for frag in OBJ.findall(t):
                try:
                    arr.append(json.loads(frag))
                except json.JSONDecodeError:
                    continue
            if arr:
                self.n_salvage += 1
            else:
                return None
        if not isinstance(arr, list):
            return None
        out = []
        for e in arr:
            if not isinstance(e, dict):
                continue
            b = e.get('bbox_2d') or e.get('bbox')
            if isinstance(b, list) and len(b) == 4:
                try:
                    out.append(([float(v) for v in b],
                                str(e.get('label', '')).lower().strip()))
                except (TypeError, ValueError):
                    continue
        return out

    def __call__(self, path, W, H):
        msgs = [{'role': 'user', 'content': [
                    {'type': 'image', self.image_key: path},
                    {'type': 'text',  'text': PROMPT}]}]
        inputs = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1536, do_sample=False,
                                      repetition_penalty=1.05)
        ans = self.proc.batch_decode(out[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0]
        parsed = self._parse(ans)
        if parsed is None:
            self.n_fail += 1
            return [], [], []

        boxes, scores, labels = [], [], []
        for b, lab in parsed:
            key = lab.split()[0] if lab else ''
            if key not in T2ID:
                continue
            if NORMALIZED:
                b = [b[0] * W / 1000, b[1] * H / 1000,
                     b[2] * W / 1000, b[3] * H / 1000]
            boxes.append(b)
            # MLLM 不输出 confidence。提示词已要求按置信度排序，
            # 这里用输出顺序赋递减分数，给 AP 计算提供排序信号。
            scores.append(max(0.01, 1.0 - 0.01 * len(scores)))
            labels.append(T2ID[key])
        return boxes, scores, labels


class YoloePredictor:
    def __init__(self, prompt_free=False, prompts=None):
        from ultralytics import YOLOE
        if prompt_free:
            self.model = YOLOE(YOLOE_PF_MODEL)
            self.pf = True
            self.name = 'YOLOE-pf'
        else:
            self.model = YOLOE(YOLOE_MODEL)
            names = prompts or ['car', 'person', 'bicycle']
            self.model.set_classes(names, self.model.get_text_pe(names))
            self.pf = False
            self.names_used = names
            self.name = 'YOLOE-' + '_'.join(n.replace(' ', '-') for n in names)
        self.n_fail = 0
        self.n_salvage = 0

    def __call__(self, path, W, H):
        r = self.model.predict(path, conf=CONF, verbose=False)[0]
        boxes, scores, labels = [], [], []
        if r.boxes is None or len(r.boxes) == 0:
            return boxes, scores, labels
        for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(),
                                   r.boxes.conf.tolist(),
                                   r.boxes.cls.tolist()):
            if not self.pf:
                cid = int(cls)               # 文本提示模式：索引即 KITTI 类别 id
                if cid > 2:
                    continue
            else:
                key = str(r.names[int(cls)]).lower().strip()
                key = key.split()[0] if key else ''
                if key not in T2ID:          # 无提示模式会发现大量非目标类别
                    continue
                cid = T2ID[key]
            boxes.append(xyxy)
            scores.append(float(conf))
            labels.append(cid)
        return boxes, scores, labels


# COCO 预训练检测器 -> KITTI 三类的跨域映射
COCO2KITTI = {
    'car': 0, 'truck': 0, 'bus': 0,
    'person': 1,
    'bicycle': 2, 'motorcycle': 2,
}


class RFDetrPredictor:
    """COCO 预训练 RF-DETR，跨域直接评测（没在 KITTI 上训过）"""

    def __init__(self, size='medium'):
        import rfdetr
        cls = {'nano': 'RFDETRNano', 'small': 'RFDETRSmall',
               'medium': 'RFDETRMedium', 'large': 'RFDETRLarge'}[size]
        self.model = getattr(rfdetr, cls)()
        self.name = 'RF-DETR-%s-coco' % size
        self.n_fail = 0
        self.n_salvage = 0
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
            self.cls_names = COCO_CLASSES
        except Exception:
            self.cls_names = None

    def __call__(self, path, W, H):
        det = self.model.predict(path, threshold=CONF)
        boxes, scores, labels = [], [], []
        xyxy = getattr(det, 'xyxy', None)
        if xyxy is None or len(xyxy) == 0:
            return boxes, scores, labels
        for bb, cid, cf in zip(xyxy, det.class_id, det.confidence):
            name = None
            if self.cls_names is not None:
                name = self.cls_names.get(int(cid)) if isinstance(self.cls_names, dict)                        else (self.cls_names[int(cid)] if int(cid) < len(self.cls_names) else None)
            if name is None:
                continue
            key = str(name).lower().strip()
            if key not in COCO2KITTI:
                continue
            boxes.append([float(v) for v in bb])
            scores.append(float(cf))
            labels.append(COCO2KITTI[key])
        return boxes, scores, labels


RTV4_REPO = os.path.expanduser('~/RT-DETRv4')


class RTDETRv4Predictor:
    """在 KITTI 上微调过的 RT-DETRv4-L（有监督基线）"""

    def __init__(self, ckpt, config=None):
        import sys
        import torch.nn as nn
        import torchvision.transforms as T
        sys.path.insert(0, RTV4_REPO)
        from engine.core import YAMLConfig

        config = config or os.path.join(
            RTV4_REPO, 'configs/rtv4/rtv4_hgnetv2_l_kitti.yml')
        cfg = YAMLConfig(config, resume=ckpt)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        state = ck['ema']['module'] if 'ema' in ck else ck['model']
        cfg.model.load_state_dict(state)

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.model = cfg.model.deploy()
                s.postprocessor = cfg.postprocessor.deploy()

            def forward(s, images, orig):
                return s.postprocessor(s.model(images), orig)

        self.device = 'cuda'
        self.model = M().to(self.device).eval()
        self.tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        self.name = 'RT-DETRv4-L-kitti'
        self.n_fail = 0
        self.n_salvage = 0
        print('已加载 %s (epoch %s)' % (os.path.basename(ckpt), ck.get('last_epoch', '?')))

    def __call__(self, path, W, H):
        im = Image.open(path).convert('RGB')
        x = self.tf(im).unsqueeze(0).to(self.device)
        orig = torch.tensor([[W, H]]).to(self.device)
        with torch.no_grad():
            labels, boxes, scores = self.model(x, orig)
        labels = labels[0].cpu().tolist()
        boxes = boxes[0].cpu().tolist()
        scores = scores[0].cpu().tolist()
        ob, osc, ol = [], [], []
        for lab, b, sc in zip(labels, boxes, scores):
            if sc < CONF:
                continue
            # 训练时 category_id 用 1/2/3（0 号空置），这里还原成 KITTI 的 0/1/2
            cid = int(lab) - 1
            if cid < 0 or cid > 2:
                continue
            ob.append([float(v) for v in b])
            osc.append(float(sc))
            ol.append(cid)
        return ob, osc, ol


class UltralyticsPredictor:
    """在 KITTI 上自训练的 ultralytics 模型（YOLO26 / RT-DETR-l），走统一评测"""

    def __init__(self, weights):
        from ultralytics import YOLO, RTDETR
        low = os.path.basename(weights).lower() + weights.lower()
        self.model = RTDETR(weights) if 'rtdetr' in low else YOLO(weights)
        self.name = os.path.basename(os.path.dirname(os.path.dirname(weights)))
        self.n_fail = 0
        self.n_salvage = 0

    def __call__(self, path, W, H):
        r = self.model.predict(path, conf=CONF, verbose=False)[0]
        ob, osc, ol = [], [], []
        if r.boxes is None or len(r.boxes) == 0:
            return ob, osc, ol
        for xyxy, cf, cl in zip(r.boxes.xyxy.tolist(),
                                r.boxes.conf.tolist(),
                                r.boxes.cls.tolist()):
            cid = int(cl)
            if cid > 2:          # kitti.yaml 只有 0/1/2
                continue
            ob.append([float(v) for v in xyxy])
            osc.append(float(cf))
            ol.append(cid)
        return ob, osc, ol


# ======================= 主流程 =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend',
                    choices=['qwen', 'yoloe', 'rfdetr', 'rtv4', 'ultra'], required=True)
    ap.add_argument('--ckpt', default=None, help='rtv4 权重路径')
    ap.add_argument('--size', default='medium', help='rfdetr 规模 nano/small/medium/large')
    ap.add_argument('--model', default=None, help='覆盖默认 HF 模型 id（仅 qwen）')
    ap.add_argument('--pf', action='store_true', help='YOLOE 无提示模式')
    ap.add_argument('--prompts', default=None,
                    help='YOLOE 文本提示，逗号分隔，顺序必须对应 Car,Pedestrian,Cyclist')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张，0=全量')
    ap.add_argument('--conf', type=float, default=None, help='覆盖默认置信度阈值')
    ap.add_argument('--tag', default=None, help='结果标签')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    global CONF
    if args.conf is not None:
        CONF = args.conf

    if args.backend == 'qwen':
        pred = QwenPredictor(args.model or QWEN_MODEL)
    elif args.backend == 'rfdetr':
        pred = RFDetrPredictor(args.size)
    elif args.backend == 'ultra':
        pred = UltralyticsPredictor(args.ckpt)
    elif args.backend == 'rtv4':
        pred = RTDETRv4Predictor(args.ckpt or os.path.expanduser(
            '~/RT-DETRv4/outputs/rtv4_l_kitti/best_stg2.pth'))
    else:
        pl = [x.strip() for x in args.prompts.split(',')] if args.prompts else None
        pred = YoloePredictor(prompt_free=args.pf, prompts=pl)

    imgs = sorted(glob.glob(os.path.join(VAL_IMG, '*')))
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        sys.exit('没找到图片，检查 VAL_IMG 路径')
    if args.tag:
        pred.name = args.tag
    print('模型 %s   图片 %d 张   conf=%s' % (pred.name, len(imgs), CONF))

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
                                  class_metrics=True,
                                  backend='faster_coco_eval')
    n_gt_missing = 0

    # --- 预测结果缓存：存下每张图的原始预测，之后可换任意指标重算，无需重跑 ---
    cache_path = os.path.expanduser('~/openvocab/preds_%s.jsonl' % pred.name)
    done = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            try:
                r = json.loads(line)
                done[r['stem']] = r
            except json.JSONDecodeError:
                continue
        print('断点续跑：已有 %d 张的缓存' % len(done))
    cache_f = open(cache_path, 'a')

    for path in tqdm(imgs, ncols=80):
        stem = os.path.splitext(os.path.basename(path))[0]
        with Image.open(path) as im:
            W, H = im.size

        if stem in done:
            r = done[stem]
            boxes, scores, labels = r['boxes'], r['scores'], r['labels']
        else:
            boxes, scores, labels = pred(path, W, H)
            cache_f.write(json.dumps({'stem': stem, 'W': W, 'H': H,
                                      'boxes': boxes, 'scores': scores,
                                      'labels': labels}) + chr(10))
            cache_f.flush()
        preds = [dict(
            boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            scores=torch.tensor(scores, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.long))]

        gb, gl = load_gt(stem, W, H)
        if gb.numel() == 0:
            n_gt_missing += 1
        tgts = [dict(boxes=gb, labels=gl)]
        metric.update(preds, tgts)

    cache_f.close()
    res = metric.compute()
    fail_rate = pred.n_fail / len(imgs) * 100

    print('\n' + '=' * 56)
    print('  模型          : %s' % pred.name)
    print('  图片数        : %d' % len(imgs))
    print('  mAP50         : %.4f' % res['map_50'].item())
    print('  mAP50-95      : %.4f' % res['map'].item())
    if 'map_per_class' in res and res['map_per_class'].numel() > 1:
        for i, v in enumerate(res['map_per_class'].tolist()):
            if i < len(NAMES):
                print('    %-12s: %.4f (mAP50-95)' % (NAMES[i], v))
    print('  JSON 解析失败 : %d 张 (%.1f%%)' % (pred.n_fail, fail_rate))
    print('  截断后救回    : %d 张' % pred.n_salvage)
    print('  无 GT 标注    : %d 张' % n_gt_missing)
    print('=' * 56)

    out = args.out or os.path.expanduser('~/openvocab/result_%s.json' % pred.name)
    payload = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in res.items()}
    payload.update(dict(model=pred.name, n_images=len(imgs),
                        normalized=NORMALIZED, json_fail=pred.n_fail,
                        json_fail_rate=fail_rate,
                        json_salvaged=pred.n_salvage))
    json.dump(payload, open(out, 'w'), indent=2, ensure_ascii=False)
    print('结果已存：%s' % out)
    print('预测缓存：%s  (可换指标重算，无需重跑)' % cache_path)


if __name__ == '__main__':
    main()
