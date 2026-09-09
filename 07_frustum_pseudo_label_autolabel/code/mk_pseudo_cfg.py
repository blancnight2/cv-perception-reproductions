# -*- coding: utf-8 -*-
"""生成 伪标签 / 真值 两套 Car 单类配置"""
import io, re, os

R = '/home/blancnight/OpenPCDet'
D = os.path.join(R, 'tools/cfgs/dataset_configs')
M = os.path.join(R, 'tools/cfgs/kitti_models')

base_ds = io.open(os.path.join(D, 'kitti_dataset.yaml'), encoding='utf-8').read()


def make_ds(name, data_path):
    s = base_ds
    s = s.replace("DATA_PATH: '../data/kitti'", "DATA_PATH: '%s'" % data_path)
    s = s.replace("filter_by_min_points: ['Car:5', 'Pedestrian:5', 'Cyclist:5']",
                  "filter_by_min_points: ['Car:5']")
    s = re.sub(r"SAMPLE_GROUPS:\s*\[[^\]]*\]", "SAMPLE_GROUPS: ['Car:15']", s)
    p = os.path.join(D, name + '.yaml')
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    return p


p1 = make_ds('kitti_pseudo_dataset', '../data/kitti_pseudo')
p2 = make_ds('kitti_gtcar_dataset', '../data/kitti')
print('数据集配置:', os.path.basename(p1), os.path.basename(p2))


def strip_anchors_to_car(s):
    """ANCHOR_GENERATOR_CONFIG 里只保留 class_name 为 Car 的那个 {...} 块"""
    lines = s.split('\n')
    out, in_anchor, buf, depth, is_car = [], False, [], 0, False
    for ln in lines:
        st = ln.strip()
        if 'ANCHOR_GENERATOR_CONFIG' in ln:
            in_anchor = True
            out.append(ln)
            continue
        if in_anchor:
            if depth == 0 and st.startswith('{'):
                depth, buf, is_car = 1, [ln], False
                continue
            if depth > 0:
                buf.append(ln)
                if "'class_name'" in st:
                    is_car = ("'Car'" in st)
                if st.startswith('}'):
                    depth = 0
                    if is_car:
                        buf[-1] = buf[-1].rstrip().rstrip(',')   # Car 成为唯一块，去尾逗号
                        out.extend(buf)
                    buf = []
                continue
            if st.startswith(']'):
                in_anchor = False
        out.append(ln)
    return '\n'.join(out)


base_m = io.open(os.path.join(M, 'pointpillar.yaml'), encoding='utf-8').read()


def make_model(name, ds):
    s = base_m
    s = s.replace("CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']", "CLASS_NAMES: ['Car']")
    s = s.replace("_BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml",
                  "_BASE_CONFIG_: cfgs/dataset_configs/%s.yaml" % ds)
    s = strip_anchors_to_car(s)
    p = os.path.join(M, name + '.yaml')
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    return p


m1 = make_model('pointpillar_car_pseudo', 'kitti_pseudo_dataset')
m2 = make_model('pointpillar_car_gt', 'kitti_gtcar_dataset')

print()
print('=== 校验 ===')
import yaml
for p in (m1, m2):
    s = io.open(p, encoding='utf-8').read()
    print('--- %s ---' % os.path.basename(p))
    for ln in s.split('\n'):
        if ln.startswith('CLASS_NAMES') or '_BASE_CONFIG_' in ln or "'class_name'" in ln:
            print('   ', ln.strip())
    try:
        y = yaml.safe_load(s)
        n = len(y['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG'])
        print('    YAML 可解析，anchor 组数 =', n)
    except Exception as e:
        print('    !! YAML 解析失败:', e)
print()
for ln in io.open(p1, encoding='utf-8').read().split('\n'):
    if 'DATA_PATH' in ln or 'filter_by_min_points' in ln or 'SAMPLE_GROUPS' in ln:
        print('   ', ln.strip())
