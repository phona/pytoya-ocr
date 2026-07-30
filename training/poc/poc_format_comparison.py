"""
PoC: 三种 OCR 输入格式对比 DeepSeek 抽取质量
  基线 — PaddleOCR-VL markdown（当前生产）
  路线① — text boxes 按 y→x 排序后拼接
  路线② — text boxes 带坐标的 JSON

用法: python3 scripts/poc_format_comparison.py
输出: 终端对比表 + /tmp/poc_format_results.json
"""

import json
import os
import re
import subprocess
import sys
import time

import requests

DEEPSEEK_API_KEY = "sk-REPLACED"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

PROMPT_RULES = """# 采购单据提取规则 (机械制造领域)
## 字段提取规则
### 1. 订单号 (po_no)
- 格式：7位数字字符串；只提取数字部分，移除"PO"等前缀。
### 2. 部门代码 (department.code)
- 从"申请/归口部门代码"字段提取。
### 3. 设备信息/用途 (invoice.usage)
- 从"申请/归口部门需求说明"字段提取。
### 4. 物品表
- 送货单的物料行：提取 name, quantity, unit, unit_price_ex_tax, unit_price_inc_tax。
- 采购单取物料名称、数量、单位、含税单价。
### 5. 金额锚定
- unit_price_* 必须能在原文中找到对应数字，严禁取整或修改。
"""

SYSTEM_PROMPT = """You are a professional data extraction system. Extract structured data from the given text following this JSON schema exactly:

{
  "invoice": {
    "po_no": "string (7 digits, extract only numbers, remove PO prefix)",
    "usage": "string (从申请/归口部门需求说明提取)",
    "invoice_date": "string (YYYY-MM-DD format)"
  },
  "department": {
    "code": "string"
  },
  "items": [{
    "name": "string",
    "quantity": "number",
    "unit": "string (KG, EA, or M)",
    "unit_price_ex_tax": "number or null",
    "unit_price_inc_tax": "number or null"
  }]
}

Rules:
""" + PROMPT_RULES + """
Only output JSON. No extra text. Prefer null over guessing."""


def call_deepseek(text_input):
    """调用 DeepSeek 抽取结构化数据。返回 parsed JSON dict。"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_input},
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(DEEPSEEK_URL, json=payload,
                          headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                          timeout=60)
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code} {r.text[:100]}"}
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return {"_error": str(e)}


def score_result(pred, gt):
    """简单字段级评分：返回 {match, total}。"""
    match = 0
    total = 0
    if not isinstance(pred, dict) or not isinstance(gt, dict):
        return {"match": 0, "total": 1}
    # invoice level fields
    for field in ["po_no", "usage", "invoice_date"]:
        p = str(pred.get("invoice", {}).get(field, "")).strip()
        g = str(gt.get("invoice", {}).get(field, "")).strip()
        total += 1
        if p and p == g:
            match += 1
    # department.code
    p = str(pred.get("department", {}).get("code", "")).strip()
    g = str(gt.get("department", {}).get("code", "")).strip()
    total += 1
    if p and p == g:
        match += 1
    # items（简化：只比对第一个 item 的 name）
    pred_items = pred.get("items", [])
    gt_items = gt.get("items", [])
    for i in range(min(len(pred_items), len(gt_items), 5)):
        for field in ["name", "quantity", "unit", "unit_price_inc_tax"]:
            p = str(pred_items[i].get(field, "")).strip()
            g = str(gt_items[i].get(field, "")).strip()
            total += 1
            if p and p == g:
                match += 1
    return {"match": match, "total": total, "score": round(match / max(total, 1) * 100, 1)}


def build_sorted_text(boxes):
    """路线①：y→x 排序后拼接文本。"""
    boxes = sorted(boxes, key=lambda b: (round(b["y"] / 20), b["x"]))
    return " ".join(b["text"] for b in boxes)


def build_json_text(boxes):
    """路线②：text boxes 带坐标的 JSON 格式文本。"""
    items = []
    for b in boxes:
        items.append(f'{{"text": "{b["text"]}", "x": {b["x"]:.0f}, "y": {b["y"]:.0f}, '
                     f'"w": {b["width"]:.0f}, "h": {b["height"]:.0f}, '
                     f'"conf": {b.get("confidence", 0):.3f}}}')
    return "[\n" + ",\n".join(items) + "\n]"


def main():
    cmd = """ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no root@47.107.92.78 "docker exec pytoya-postgres psql -U postgres -d pytoya -tAc \\"SELECT id, filename, extracted_data::text, left(ocr_result->'pages'->0->>'text', 2000) FROM manifests WHERE id IN (50, 51, 52, 53, 54) ORDER BY id;\\"" """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    lines = result.stdout.strip().split('\n')

    preds = json.load(open('pytoya-ocr/ft_predictions_v8.json'))
    import re

    results = []

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split('|', 3)
        if len(parts) < 4:
            continue
        mid, fn_raw, extracted_str, pvl_md = [p.strip() for p in parts]
        gt = json.loads(extracted_str)
        uuid_m = re.search(r'([a-f0-9\-]{36})', fn_raw)
        if not uuid_m:
            continue
        prefix = uuid_m.group(1)[:12]
        img_fn = f"m_{prefix}_p1.png"
        all_boxes = preds.get(img_fn, [])
        boxes = [b for b in all_boxes if b.get("text", "").strip()]
        if not boxes:
            continue

        print(f"\n{'='*70}")
        print(f"M{mid} → {img_fn}  ({len(boxes)} boxes)")
        print('='*70)

        # 构建三种输入
        input_markdown = pvl_md[:1500]  # PaddleOCR-VL markdown（截短）
        input_sorted_text = build_sorted_text(boxes)[:2000]
        input_json = build_json_text(boxes)[:2000]

        # 依次调用 DeepSeek
        res_md = call_deepseek(input_markdown)
        res_r1 = call_deepseek(input_sorted_text)
        res_r2 = call_deepseek(input_json)

        # 评分
        s_md = score_result(res_md, gt)
        s_r1 = score_result(res_r1, gt)
        s_r2 = score_result(res_r2, gt)

        row = {
            "mid": mid, "img": img_fn,
            "scores": {"markdown": s_md, "route1_sorted": s_r1, "route2_json": s_r2},
            "outputs": {"markdown": res_md, "route1": res_r1, "route2": res_r2},
        }
        results.append(row)

        print(f"  {'格式':<20} {'字段匹配':>15} {'总分':>8}")
        print(f"  {'PaddleOCR-VL markdown':<20} {s_md['match']:>4}/{s_md['total']:<4}      {s_md['score']:>6.1f}%")
        print(f"  {'路线① 排序文本':<20} {s_r1['match']:>4}/{s_r1['total']:<4}      {s_r1['score']:>6.1f}%")
        print(f"  {'路线② JSON+坐标':<20} {s_r2['match']:>4}/{s_r2['total']:<4}      {s_r2['score']:>6.1f}%")

    # 汇总
    if results:
        print(f"\n{'='*70}")
        print(f"汇总（{len(results)} 页）")
        print('='*70)
        print(f"{'M':>4} {'markdown':>10} {'路线①':>10} {'路线②':>10}")
        for r in results:
            print(f"{r['mid']:>4} {r['scores']['markdown']['score']:>9.1f}% "
                  f"{r['scores']['route1_sorted']['score']:>9.1f}% "
                  f"{r['scores']['route2_json']['score']:>9.1f}%")
        avg_md = sum(r['scores']['markdown']['score'] for r in results) / len(results)
        avg_r1 = sum(r['scores']['route1_sorted']['score'] for r in results) / len(results)
        avg_r2 = sum(r['scores']['route2_json']['score'] for r in results) / len(results)
        print(f"{'平均':>4} {avg_md:>9.1f}% {avg_r1:>9.1f}% {avg_r2:>9.1f}%")

    json.dump(results, open("/tmp/poc_format_results.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n明细 → /tmp/poc_format_results.json")


if __name__ == "__main__":
    main()
