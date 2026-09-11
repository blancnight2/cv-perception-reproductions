# -*- coding: utf-8 -*-
"""把端到端时延换算成行车语言：这段时间车开出去多远、盲区多大。

数据来自 03 项目实测的 summary.json（1501 帧，排除 100 帧预热）。
"""
import json
import sys
import io

SPEEDS = [(30, '城区'), (60, '快速路'), (90, '高速'), (120, '高速限速')]


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    d = json.load(open(src, encoding='utf-8'))
    e = d['end_to_end']
    L = []
    w = L.append

    w('## 时延的行车含义\n')
    w('推理快慢本身没有意义，要问的是**这段时间车开出去多远**。')
    w('下表由实测端到端时延（%d 帧，排除 %d 帧预热）直接换算：\n'
      % (e['count'], d['warmup_frames_excluded']))

    w('| 车速 | | p50 %.2f ms | p95 %.2f ms | p99 %.2f ms | 最坏 %.2f ms |'
      % (e['p50_ms'], e['p95_ms'], e['p99_ms'], e['max_ms']))
    w('|---|---|---|---|---|---|')
    for kmh, name in SPEEDS:
        v = kmh / 3.6
        cells = ' | '.join('%.1f cm' % (v * e[k] / 1000 * 100)
                           for k in ('p50_ms', 'p95_ms', 'p99_ms', 'max_ms'))
        w('| %d km/h | %s | %s |' % (kmh, name, cells))

    w('\n> 最坏一帧 %.2f ms，在 120 km/h 下对应 **%.1f cm** 的位移——'
      '这是**单帧最大盲区**，也是为什么部署要看 p99/max 而不是均值。'
      % (e['max_ms'], (120 / 3.6) * e['max_ms'] / 1000 * 100))

    w('\n### 一个更该问的问题：瓶颈根本不在推理\n')
    fps_cam = 30.0
    interval = 1000.0 / fps_cam
    stale = interval + e['p50_ms']
    w('车载相机通常 **%d FPS**，即每帧间隔 **%.1f ms**。' % (int(fps_cam), interval))
    w('一个目标从"被拍到"到"结果可用"的**总时延 = 帧间隔 + 处理时延 = %.1f + %.2f = %.1f ms**，'
      % (interval, e['p50_ms'], stale))
    w('其中处理只占 **%.1f%%**。\n' % (100 * e['p50_ms'] / stale))
    w('| 车速 | 仅处理时延 | 帧间隔+处理 | 处理占比 |')
    w('|---|---|---|---|')
    for kmh, _ in SPEEDS:
        v = kmh / 3.6
        w('| %d km/h | %.1f cm | **%.1f cm** | %.1f%% |'
          % (kmh, v * e['p50_ms'] / 1000 * 100, v * stale / 1000 * 100,
             100 * e['p50_ms'] / stale))
    w('\n> **结论：在 30 FPS 相机下，把推理从 %.2f ms 再优化到 3 ms，总盲区只从 %.1f cm 降到 %.1f cm（约 %.0f%%）。**'
      % (e['p50_ms'], (60 / 3.6) * stale / 1000 * 100,
         (60 / 3.6) * (interval + 3) / 1000 * 100,
         100 * (1 - (interval + 3) / stale)))
    w('> 桌面 GPU 上继续榨推理是**优化错了地方**——真正的余量在相机帧率和端侧算力。')
    w('> 这也正是下一步要上边缘设备的理由：Jetson 上处理时延会涨一个量级，那时它才成为瓶颈。\n')

    w('### 时延构成（均值）\n')
    w('| 阶段 | 均值 ms | 占比 |')
    w('|---|---|---|')
    tot = e['mean_ms']
    for k, nm in (('decode', '视频解码'), ('detect_nms_track', '检测+NMS+ByteTrack'),
                  ('render', '渲染绘制')):
        if k in d:
            w('| %s | %.3f | %.1f%% |' % (nm, d[k]['mean_ms'], 100 * d[k]['mean_ms'] / tot))
    w('| **端到端** | **%.3f** | 100%% |' % tot)
    w('\n> 检测+跟踪占 **%.1f%%**，解码和渲染合计 **%.1f%%**——'
      '端侧优化要连解码一起算，只优化模型是不够的。'
      % (100 * d['detect_nms_track']['mean_ms'] / tot,
         100 * (d['decode']['mean_ms'] + d['render']['mean_ms']) / tot))

    txt = '\n'.join(L) + '\n'
    print(txt)
    if out:
        io.open(out, 'w', encoding='utf-8', newline='\n').write(txt)
        print('已写出 -> %s' % out)


if __name__ == '__main__':
    main()
