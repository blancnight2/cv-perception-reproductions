# -*- coding: utf-8 -*-
"""
修 eval_zeroshot.py 的 QwenPredictor 加载方式（本地和 AutoDL 都要打）。

原代码：
    self.model = cls.from_pretrained(
        model_id, quantization_config=bnb, device_map='auto').eval()

问题：device_map='auto' 在 Qwen3.6-35B-A3B 上估算错误，把层甩到 CPU，
      bnb 4-bit 直接拒绝："Some modules are dispatched on the CPU or the disk"。
      且没传 dtype，未量化部分可能按 fp32 估，估值虚高。

改为：device_map={'': 0} 全放 0 号卡 + 显式 dtype=bfloat16 + low_cpu_mem_usage。
      （已用 diag_fit.py 确认 4-bit 峰值约 23.9GB < 5090 的 33.7GB）

用法：python patch_qwen_loader.py [eval_zeroshot.py 的路径]
"""
import sys, os, re, shutil

P = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/openvocab/eval_zeroshot.py')
src = open(P, encoding='utf-8').read()

OLD = """        self.model = cls.from_pretrained(
            model_id, quantization_config=bnb, device_map='auto').eval()"""

NEW = """        # device_map='auto' 在 Qwen3.6-35B-A3B 上估算错误会甩层到 CPU，
        # 导致 bnb 4-bit 报 "Some modules are dispatched on the CPU or the disk"。
        # 显存已核算够用 -> 全放 0 号卡，绕开自动估算；dtype 必须显式给。
        _kw = dict(quantization_config=bnb, device_map={'': 0},
                   low_cpu_mem_usage=True)
        try:
            self.model = cls.from_pretrained(
                model_id, dtype=torch.bfloat16, **_kw).eval()
        except TypeError:                      # 旧版 transformers 用 torch_dtype
            self.model = cls.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, **_kw).eval()"""

if NEW.strip()[:40] in src:
    print('已经打过补丁，跳过')
    sys.exit(0)

if OLD not in src:
    print('!! 没找到目标代码块，请手动改。当前 from_pretrained 附近：')
    for m in re.finditer(r'.*from_pretrained.*', src):
        print('   ', m.group(0).strip())
    sys.exit(1)

shutil.copyfile(P, P + '.bak')
open(P, 'w', encoding='utf-8').write(src.replace(OLD, NEW))
print('已打补丁：%s' % P)
print('原文件备份：%s.bak' % P)

# 确认 torch 已在文件顶部 import（NEW 里用到 torch.bfloat16）
head = src[:2000]
print('torch 已导入：%s' % ('是' if re.search(r'^import torch', head, re.M) else '否 —— 需手动加'))
