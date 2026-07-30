"""
PoC: 带行上下文的 VLM 裁切识别 vs 裸 crop 识别

假设：给 VLM 该框所在的表格行信息（字段类型 + 邻居字段值）
→ 识别准确率显著高于裸裁图识别。

方法：
  1. 从 corrections 里抽 conf<0.85 的难框
  2. 对每个框，从 v8 预测里找到它所在的行（按 y 聚类）
  3. 提取该行所有非空框的文本作为「邻居上下文」
  4. 对每个框跑两次 VLM：
     a. 裸提示词（基线）
     b. 上下文增强提示词（含字段类型推测 + 邻居值）
  5. 比较两组准确率。

输出：逐框双行对比表 + 统计。

用法: python3 scripts/poc_vlm_with_context.py [--model MODEL]
      默认模型 Qwen/Qwen3-VL-8B-Instruct
"""

import base64
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from collections import Counter

import requests
from PIL import Image
import sys

ROOT = "/mnt/e/pytoya-workspace"
CORR = os.path.join(ROOT, "data/corrections.json")
PRED = os.path.join(ROOT, "pytoya-ocr/ft_predictions_v8.json")
IMG_DIR = os.path.join(ROOT, "data/pages_all/images")
OUT = "/tmp/poc_context_results.json"
CROP_DIR = "/tmp/poc_crops"
os.makedirs(CROP_DIR, exist_ok=True)

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "20"))
MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MAX_WORKERS = 3
KEY = os.environ.get("SILICONFLOW_API_KEY", "")
BASE = "https://api.siliconflow.cn/v1/chat/completions"

if not KEY:
    for p in ["/tmp/cred.json"]:
        if os.path.exists(p):
            KEY = json.load(open(p))["apiKey"]
            break
if not KEY:
    print("请设置 SILICONFLOW_API_KEY 环境变量", file=sys.stderr)
    sys.exit(1)


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return prev[-1]
def lev_sim(a, b):
    a, b = norm(a), norm(b)
    return 1 - lev(a, b) / max(len(a), len(b)) if max(len(a), len(b)) else 0

# ---------------------------------------------------------------------------
# Step 1: 抽难框 + 匹配 v8 框
# ---------------------------------------------------------------------------
def load_boxes(n):
    corr = json.load(open(CORR))
    edits = [c for c in corr if c["type"] == "改文字"]
    edits.sort(key=lambda c: c.get("conf", 1))  # conf 最低的最难
    preds = json.load(open(PRED))
    boxes = []
    for c in edits:
        fn, gt = c["image"], c.get("final", "")
        pboxes = preds.get(fn, [])
        if not gt or not pboxes: continue
        img = Image.open(os.path.join(IMG_DIR, fn))
        W, H = img.size
        cx_px, cy_px = c["cx"] * W, c["cy"] * H
        best, bd = None, float("inf")
        for pb in pboxes:
            pcx = pb["x"] + pb["width"] / 2
            pcy = pb["y"] + pb["height"] / 2
            d = (pcx - cx_px)**2 + (pcy - cy_px)**2
            if d < bd: bd, best = d, pb
        if best is None or bd > max(W, H) / 10: continue
        x1, y1 = max(0, int(best["x"])), max(0, int(best["y"]))
        x2, y2 = min(W, int(best["x"] + best["width"])), min(H, int(best["y"] + best["height"]))
        if x2 - x1 < 3 or y2 - y1 < 3: continue
        crop = img.crop((x1, y1, x2, y2))
        crop_path = os.path.join(CROP_DIR, f"{fn.replace('.png','')}_{x1}_{y1}.png")
        crop.save(crop_path)

        # 找同行邻居（y 中心相近的框）
        cy = best["y"] + best["height"] / 2
        row_mates = [pb for pb in pboxes if abs((pb["y"] + pb["height"] / 2) - cy) < cy * 0.15 and pb != best]
        row_mates.sort(key=lambda b: b["x"])
        neighbor_texts = []
        for rm in row_mates[:4]:
            t = rm.get("text", "").strip()
            if t and norm(t) != norm(best.get("text", "")): neighbor_texts.append(t)
        context_hint = "；".join(neighbor_texts) if neighbor_texts else ""
        # 字段类型推测
        field_type = "表格文字"
        all_text = "".join(b.get("text", "") for b in pboxes)
        if any(kw in all_text for kw in ["单价", "金额", "数量"]):
            # 看这个框的位置来猜
            bx_ratio = (best["x"] + best["width"] / 2) / W
            by_ratio = (best["y"] + best["height"] / 2) / H
            # 送货单表格：列大致位置
            # 序号0-12%, 名称14-35%, 单位38-46%, 数量48-56%, 单价58-66%, 金额68-76%, 备注78-90%
            col_map = [(0.00,0.12,"序号"),(0.14,0.35,"名称"),(0.38,0.46,"单位"),
                       (0.48,0.56,"数量"),(0.58,0.66,"含税单价"),(0.68,0.76,"金额"),(0.78,0.90,"备注")]
            ft = "表格文字"
            for cs, ce, cl in col_map:
                if cs <= bx_ratio <= ce and by_ratio > 0.15: ft = cl; break
            field_type = ft

        boxes.append({
            "fn": fn, "crop": crop_path, "gt": gt,
            "field_type": field_type, "context": context_hint,
            "v8_text": best.get("text", ""), "v8_conf": round(best.get("confidence", 0), 3),
        })
        if len(boxes) >= n: break
    return boxes

# ---------------------------------------------------------------------------
# Step 2: VLM 调用（双提示词）
# ---------------------------------------------------------------------------
PROMPT_BARE = "请准确识别图片中的文字，只输出识别结果，不要解释、不要前后缀、不要markdown。"
PROMPT_TEMPLATE = "以下是采购单表格中一个「{ft}」字段的裁切图。该行其他字段的识别结果为：「{ctx}」。请识别图中的文字，只输出识别结果，不要解释。"

def call_vlm(prompt, crop_path, timeout=30):
    with open(crop_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}], "max_tokens": 256, "temperature": 0}
    try:
        r = requests.post(BASE, json=payload,
                          headers={"Authorization": f"Bearer {KEY}"}, timeout=timeout)
        return r.json()["choices"][0]["message"]["content"].strip() if r.status_code == 200 else ""
    except: return ""

def main():
    boxes = load_boxes(SAMPLE_SIZE)
    print(f"采样 {len(boxes)} 个手写难框\n")

    # 跑两轮提示词
    for phase_name, prompt_fn in [("裸识别", lambda b: PROMPT_BARE),
                                    ("上下文增强", lambda b: PROMPT_TEMPLATE.format(
                                        ft=b["field_type"], ctx=b["context"]))]:
        print(f"\n=== {phase_name} ===")
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(call_vlm, prompt_fn(b), b["crop"]): b for b in boxes}
            for fu in cf.as_completed(futs):
                b = futs[fu]
                key = "vlm_bare" if "裸" in phase_name else "vlm_context"
                b[key] = fu.result()
        print(f"耗时 {time.time() - t0:.0f}s")

    # 对比表
    print(f"\n{'':>3} {'字段类型':<10} {'GT':<14} {'裸VLM':<18} {'上下文VLM':<18} {'邻居':<20} {'裸判':<6} {'上下判':<6}")
    print("-" * 100)
    stats_bare = Counter()
    stats_ctx = Counter()

    for i, b in enumerate(boxes, 1):
        gt = b["gt"][:12]
        bare = (b.get("vlm_bare", "") or "")[:14]
        ctx = (b.get("vlm_context", "") or "")[:14]
        ctx_nei = b["context"][:18]
        s_bare = lev_sim(b.get("vlm_bare", ""), b["gt"])
        s_ctx = lev_sim(b.get("vlm_context", ""), b["gt"])
        # 分类
        def classify(s, v):
            if not v: return "💤"
            if s >= 1.0: return "✅"
            if s >= 0.85: return "⚠️"
            if len(v) >= 8 and s < 0.3: return "⚡"
            return "❌"
        lb = classify(s_bare, b.get("vlm_bare", ""))
        lc = classify(s_ctx, b.get("vlm_context", ""))
        stats_bare[lb] += 1; stats_ctx[lc] += 1
        print(f"{i:>3} {b['field_type']:<10} {gt:<14} {bare:<18} {ctx:<18} {ctx_nei:<20} {lb:<6} {lc:<6}")

    # 汇总
    print(f"\n{'':>26} {'裸VLM':>20} {'上下文VLM':>20}")
    for lbl in ["✅救回","⚠️接近","❌错","⚡幻觉","💤空"]:
        k = {"✅救回":"✅","⚠️接近":"⚠️","❌错":"❌","⚡幻觉":"⚡","💤空":"💤"}[lbl]
        nb = stats_bare.get(k, 0); nc = stats_ctx.get(k, 0)
        print(f"  {lbl:>20}: {nb:>5}({nb/len(boxes)*100:.0f}%)       {nc:>5}({nc/len(boxes)*100:.0f}%)")
    good_bare = stats_bare["✅"] + stats_bare["⚠️"]
    good_ctx = stats_ctx["✅"] + stats_ctx["⚠️"]
    print(f"  {'救回+接近':>20}: {good_bare}/{len(boxes)}={good_bare/len(boxes)*100:.0f}%       {good_ctx}/{len(boxes)}={good_ctx/len(boxes)*100:.0f}%")
    print(f"\n  上下文提升: {good_ctx - good_bare}/{len(boxes)} = +{(good_ctx-good_bare)/len(boxes)*100:.0f}%")

    json.dump(boxes, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n明细 → {OUT}")


if __name__ == "__main__":
    main()
