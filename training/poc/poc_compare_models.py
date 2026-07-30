"""
多模型横向对比 PoC：同 29 个难框 crop，同时跑 8B / 32B / 72B，比较输出。

用法:
  python3 scripts/poc_compare_models.py

前置: 已有 /tmp/poc_results.json（第一步 PoC 的输出，含 crop 路径 + 人标真值）
输出: 终端打印三模型对比表 + 统计
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

logging = None  # suppress PIL notice
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ROOT = "/mnt/e/pytoya-workspace"
CORR = os.path.join(ROOT, "data/corrections.json")
PRED = os.path.join(ROOT, "pytoya-ocr/ft_predictions_v8.json")
IMG_DIR = os.path.join(ROOT, "data/pages_all/images")

API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
if not API_KEY:
    print("请设置 SILICONFLOW_API_KEY", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"

MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-72B-Instruct",
]

PROMPT = "请准确识别图片中的文字，只输出识别结果，不要解释、不要前后缀、不要markdown。"
MAX_WORKERS = 3


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
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return prev[-1]


def lev_sim(a, b):
    a, b = norm(a), norm(b)
    return 1 - lev(a, b) / max(len(a), len(b)) if max(len(a), len(b)) else 0.0


def call_vlm(model, crop_path, timeout=45):
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
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
        return "" if r.status_code != 200 else r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def main():
    # 读取第一步 PoC 的结果
    src = json.load(open("/tmp/poc_results.json"))
    print(f"加载 {len(src)} 条难框 (来自第一步 PoC)")

    rows = []
    for b in src:
        crop = b.get("crop", "")
        if not crop or not os.path.exists(crop):
            continue
        rows.append({
            "fn": b["fn"],
            "crop": crop,
            "v8_text": b["v8_text"],
            "v8_conf": b["v8_conf"],
            "gt": b["human_gt"],
            "vlm_8B": b.get("vlm_text", ""),  # 第一步已经跑了 8B
        })

    print(f"可用 crop 数: {len(rows)}")

    # 对每个 crop，跑 32B 和 72B
    for midx, model in enumerate(MODELS[1:], 1):  # 32B, 72B
        key = f"vlm_{model.split('/')[-1]}"  # "VLM-32B-Instruct"
        print(f"\n调 {model} ({len(rows)} 次)...", end=" ", flush=True)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut_map = {ex.submit(call_vlm, model, r["crop"]): r for r in rows}
            for fu in cf.as_completed(fut_map):
                r = fut_map[fu]
                r[key] = fu.result()
        print(f"{time.time() - t0:.0f}s")

    # 打分
    def score(vlm_text, gt):
        if not vlm_text:
            return "💤空", 0
        s = lev_sim(vlm_text, gt)
        if s >= 1.0:
            return "✅", s
        if s >= 0.85:
            return "⚠️", s
        # 幻觉检测
        if len(vlm_text) >= 8 and s < 0.3:
            return "⚡", s
        return "❌", s

    for r in rows:
        for col in ["vlm_8B", "vlm_Qwen3-VL-32B-Instruct", "vlm_Qwen3-VL-72B-Instruct"]:
            if col in r:
                lbl, sim = score(r[col], r["gt"])
                r[col + "_label"] = lbl
                r[col + "_sim"] = round(sim, 3)
            else:
                r[col + "_label"] = "💤"
                r[col + "_sim"] = 0

    # 统计
    cols = ["vlm_8B", "vlm_Qwen3-VL-32B-Instruct", "vlm_Qwen3-VL-72B-Instruct"]
    labels_display = ["8B", "32B", "72B"]
    stats = {}
    for li, col in enumerate(cols[:3]):
        counter = Counter(r[col + "_label"] for r in rows if col in r)
        stats[labels_display[li]] = counter

    # 打印汇总表
    print(f"\n{'':>3} {'图片':<28} {'v8_text':<14} {'GT':<18} {'8B':<20} {'32B':<20} {'72B':<20}")
    print("-" * 125)
    for i, r in enumerate(rows, 1):
        fn = r["fn"][:26]
        v8t = r["v8_text"][:12]
        gt = r["gt"][:16]
        outs = []
        for col in cols:
            txt = (r.get(col, "") or "")[:14]
            lbl = r.get(col + "_label", "")
            sim = r.get(col + "_sim", 0)
            outs.append(f"{txt}({lbl}{sim})" if lbl else f"{txt:18s}")
        print(f"{i:>3} {fn:<28} {v8t:<14} {gt:<18} {outs[0]:<20} {outs[1]:<20} {outs[2]:<20}")

    print("\n=== 各模型汇总 ===")
    for lbl in labels_display:
        cnt = stats[lbl]
        total = sum(cnt.values())
        good = cnt.get("✅", 0) + cnt.get("⚠️", 0)
        hallu = cnt.get("⚡", 0)
        ratio = good / total * 100 if total else 0
        print(f"  {lbl:>6}: ✅+⚠️={good}/{total}={ratio:.0f}%  ⚡幻觉={hallu}  ({dict(cnt)})")

    # 保存联合结果
    json.dump(rows, open("/tmp/poc_multimodel.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n联合结果已写入 /tmp/poc_multimodel.json")


if __name__ == "__main__":
    main()
