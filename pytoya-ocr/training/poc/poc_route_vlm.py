"""
PoC: PaddleOCR 低置信框裁切 → VLM 兜底 — 链路效果验证

用法:
  # 默认 8B
  python3 scripts/poc_route_vlm.py
  # 调 32B
  VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct python3 scripts/poc_route_vlm.py
  # 调不同 baseUrl（如 GLM / GPT）
  VLM_MODEL=… VLM_BASE=https://… python3 scripts/poc_route_vlm.py

前置（第一次用须设 key）:
  export SILICONFLOW_API_KEY=sk-…
  或脚本自动从 /tmp/cred.json 读（上一轮会话若存过）

输出:
  - /tmp/poc_crops/             裁切的 crop 图
  - /tmp/poc_results.json       逐框结果
  - 终端打印横向对比表 + 统计
"""

import base64
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections import Counter

import requests

logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# 环境 / 常量
# ---------------------------------------------------------------------------
ROOT = "/mnt/e/pytoya-workspace"
CORR = os.path.join(ROOT, "data/corrections.json")
PRED = os.path.join(ROOT, "pytoya-ocr/ft_predictions_v8.json")
IMG_DIR = os.path.join(ROOT, "data/pages_all/images")
OUT = "/tmp/poc_results.json"
CROP_DIR = "/tmp/poc_crops"
os.makedirs(CROP_DIR, exist_ok=True)

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "40"))
VLM_MODEL = os.environ.get(
    "VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"
)
VLM_BASE = os.environ.get("VLM_BASE", "https://api.siliconflow.cn/v1")
MAX_WORKERS = int(os.environ.get("VLM_WORKERS", "5"))
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.85"))

# ---------------------------------------------------------------------------
# 凭据
# ---------------------------------------------------------------------------
def get_vlm_key():
    if key := os.environ.get("SILICONFLOW_API_KEY"):
        return key
    cred_path = "/tmp/cred.json"
    if os.path.exists(cred_path):
        return json.load(open(cred_path))["apiKey"]
    print(
        "\n请设置 SILICONFLOW_API_KEY 环境变量（或重做 /tmp/cred.json），"
        "例如从 pytoya 库 extractors 表获取。\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
KANA = re.compile(r"[぀-ヿ]")

def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def lev(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)
            )
        prev = cur
    return prev[-1]


def lev_sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 1 - lev(a, b) / max(len(a), len(b))

# ---------------------------------------------------------------------------
# Step 1 — 从 corrections 抽「手写嫌疑」难框
# ---------------------------------------------------------------------------
def sample_hard_boxes(n):
    corr = json.load(open(CORR))
    edits = [c for c in corr if c["type"] == "改文字"]

    def hand_score(c):
        """分数越高 = 越可能是潦草手写/难框（优先抽）。"""
        s = 0
        if KANA.search(c.get("model", "") or ""):
            s += 5
        if c.get("conf", 1.0) < 0.7:
            s += 3
        if c.get("conf", 1.0) < CONF_THRESH:
            s += 2
        m, f = (c.get("model", "") or ""), (c.get("final", "") or "")
        if lev_sim(m, f) < 0.4:
            s += 2
        if len(f) <= 4 and len(m) <= 4 and m != f:
            s += 1  # 短字段纠错
        return s

    edits.sort(key=hand_score, reverse=True)
    return edits[:n]

# ---------------------------------------------------------------------------
# Step 2 — 匹配 v8 预测框、裁 crop
# ---------------------------------------------------------------------------
from PIL import Image


def match_and_crop(c, preds):
    """
    对一条 correction（改文字），匹配 v8 预测框并裁图。
    返回 dict 或 None（匹配失败）。
    """
    fn = c.get("image", "")
    pboxes = preds.get(fn, [])
    if not pboxes:
        return None
    img_path = os.path.join(IMG_DIR, fn)
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path)
    W, H = img.size
    cx_px = c["cx"] * W
    cy_px = c["cy"] * H

    # 找最近框（欧氏距离）
    best = None
    best_d = float("inf")
    for pb in pboxes:
        pb_cx = pb["x"] + pb["width"] / 2
        pb_cy = pb["y"] + pb["height"] / 2
        d = (pb_cx - cx_px) ** 2 + (pb_cy - cy_px) ** 2
        if d < best_d:
            best_d = d
            best = pb
    if best_d > max(W, H) / 10:  # 太远不配
        return None
    # 裁 crop
    x1 = max(0, int(best["x"]))
    y1 = max(0, int(best["y"]))
    x2 = min(W, int(best["x"] + best["width"]))
    y2 = min(H, int(best["y"] + best["height"]))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = img.crop((x1, y1, x2, y2))
    crop_hash = hashlib.md5(f"{fn}_{x1}_{y1}_{x2}_{y2}".encode()).hexdigest()[:8]
    crop_name = f"{crop_hash}.png"
    crop_path = os.path.join(CROP_DIR, crop_name)
    crop.save(crop_path)
    return {
        "fn": fn,
        "crop": crop_path,
        "v8_conf": round(best.get("confidence", 0), 4),
        "v8_text": best.get("text", ""),
        "human_gt": c.get("final", ""),
        "v6_text": c.get("model", "") or "",
    }

# ---------------------------------------------------------------------------
# Step 3 — 调 VLM
# ---------------------------------------------------------------------------
PROMPT = "请准确识别图片中的文字，只输出识别结果，不要解释、不要前后缀、不要markdown。"


def call_vlm(api_url, model, key, crop_path, timeout=30):
    with open(crop_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return f"[ERR HTTP {r.status_code}]"
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERR {e}]"

# ---------------------------------------------------------------------------
# Step 4 — 比对 + 裁决
# ---------------------------------------------------------------------------
def verdict(vlm_text, human_gt):
    """返回 (label, edit_sim)。label∈{'✅救回','⚠️接近','❌错','❌幻觉','💤空','⚡ERR'}"""
    v = (vlm_text or "").strip()
    g = (human_gt or "").strip()
    if not v:
        return ("💤空", 0)
    if v.startswith("[ERR"):
        return ("⚡ERR", 0)
    if norm(v) == norm(g):
        return ("✅救回", 1.0)
    sim = lev_sim(v, g)
    if sim >= 0.85:
        return ("⚠️接近", sim)
    # 幻觉检测：VLM 输出了和 gt 完全不相干的词组（长度超 6 且 sim < 0.2）
    if len(v) >= 8 and sim < 0.3:
        return ("❌幻觉", sim)
    return ("❌错", sim)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"参数: model={VLM_MODEL}, sample={SAMPLE_SIZE}, conf_thresh={CONF_THRESH}, workers={MAX_WORKERS}")
    print()
    api_key = get_vlm_key()
    api_url = VLM_BASE.rstrip("/") + "/chat/completions"

    # 1. 抽难框
    samples = sample_hard_boxes(SAMPLE_SIZE)
    print(f"从 corrections 抽了 {len(samples)} 条难框")

    # 2. 匹配 + 裁切
    preds = json.load(open(PRED))
    boxes = []
    for c in samples:
        b = match_and_crop(c, preds)
        if b:
            boxes.append(b)
    print(f"成功匹配+裁切 {len(boxes)} 个 crop → {CROP_DIR}/")
    if not boxes:
        sys.exit(1)
    # 3. 调 VLM
    st = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(call_vlm, api_url, VLM_MODEL, api_key, b["crop"]): b for b in boxes}
        for fu in cf.as_completed(fut_map):
            b = fut_map[fu]
            vlm_out = fu.result()
            label, sim = verdict(vlm_out, b["human_gt"])
            b["vlm_text"] = vlm_out
            b["verdict"] = label
            b["sim"] = round(sim, 3)
            results.append(b)

    elapsed = time.time() - st
    # 4. 横向对比表
    print(f"\nVLM 调用完毕 ({elapsed:.0f}s)，{len(results)} 条结果\n")
    # 排序：救回优先显示，err 最后
    order = {"✅救回": 0, "⚠️接近": 1, "❌错": 2, "❌幻觉": 3, "💤空": 4, "⚡ERR": 5}
    results.sort(key=lambda r: (order.get(r["verdict"], 9), -r["sim"]))

    print(f"{'序':>2} {'图片':<28} {'v8_conf':>7} {'v8_text':<16} {'→VLM':<16} {'人标':<20} {'裁决':<10} {'相似':>5}")
    print("-" * 110)
    for i, b in enumerate(results, 1):
        fn = b["fn"][:26]
        v8t = b["v8_text"][:14]
        vlm = (b.get("vlm_text") or "")[:14]
        gt = b["human_gt"][:18]
        print(f"{i:>2} {fn:<28} {b['v8_conf']:>7.3f} {v8t:<16} {vlm:<16} {gt:<20} {b['verdict']:<10} {b['sim']:.3f}")

    # 5. 统计
    print("\n=== 汇总 ===")
    counter = Counter(r["verdict"] for r in results)
    for label in ["✅救回", "⚠️接近", "❌错", "❌幻觉", "💤空", "⚡ERR"]:
        print(f"  {label}: {counter[label]}")
    total = max(len(results), 1)
    print(f"\n人可接受指标（救回 + 接近）: {counter['✅救回'] + counter['⚠️接近']}/{len(results)} = "
          f"{(counter['✅救回'] + counter['⚠️接近']) / total * 100:.1f}%")
    print(f"PaddleOCR 在这些难框上的自动准确率: 0.0%（因为抽的就是它读错的）")
    vlm_only_correct = counter["✅救回"] + counter["⚠️接近"]
    print(f"VLM 兜底后准确率: {vlm_only_correct}/{len(results)} = "
          f"{vlm_only_correct / total * 100:.1f}%")
    print(f"幻觉风险: {counter['❌幻觉']}/{total} = "
          f"{counter['❌幻觉'] / total * 100:.1f}%（这些值直接过的话会污染数据）")

    # 6. 保存详细结果
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {OUT}")


if __name__ == "__main__":
    main()
