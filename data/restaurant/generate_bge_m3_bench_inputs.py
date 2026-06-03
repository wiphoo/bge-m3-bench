#!/usr/bin/env python3
"""Generate restaurant benchmark inputs for wiphoo/bge-m3-bench.

The benchmark repo accepts `--texts inputs.txt`: one raw text per line.
This script creates deterministic Thai/English restaurant texts in fixed-ish
length buckets for embedding performance tests.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

SEED = 20260604

CUISINE_MAP = {
    "pizza_restaurant": ("pizza", "พิซซ่า", ["Margherita pizza", "pepperoni pizza", "garlic bread", "pasta", "Caesar salad"]),
    "japanese_restaurant": ("Japanese", "อาหารญี่ปุ่น", ["ramen", "sushi set", "gyoza", "katsu curry", "miso soup"]),
    "thai_restaurant": ("Thai", "อาหารไทย", ["pad thai", "green curry", "tom yum", "stir-fried basil", "mango sticky rice"]),
    "chinese_restaurant": ("Chinese", "อาหารจีน", ["dim sum", "roast duck", "fried rice", "wonton soup", "mapo tofu"]),
    "cafe": ("cafe", "คาเฟ่", ["iced latte", "americano", "croissant", "cake", "matcha latte"]),
    "coffee_shop": ("coffee", "กาแฟ", ["espresso", "cold brew", "latte", "cappuccino", "brownie"]),
    "fast_food_restaurant": ("fast food", "ฟาสต์ฟู้ด", ["burger", "fried chicken", "fries", "nuggets", "soft drink"]),
    "seafood_restaurant": ("seafood", "อาหารทะเล", ["grilled prawns", "steamed fish", "crab curry", "fried squid", "spicy seafood soup"]),
    "restaurant": ("casual dining", "ร้านอาหารทั่วไป", ["signature rice bowl", "noodle soup", "grilled chicken", "fried rice", "house salad"]),
}

AREAS = [
    ("Sathorn", "สาทร"), ("Siam", "สยาม"), ("Thonglor", "ทองหล่อ"), ("Bang Na", "บางนา"),
    ("Rama 9", "พระราม 9"), ("Ari", "อารีย์"), ("Silom", "สีลม"), ("Phra Khanong", "พระโขนง"),
    ("Bangkok", "กรุงเทพ"), ("Ratchada", "รัชดา"),
]
TASTES = [("spicy", "เผ็ด"), ("mild", "รสไม่จัด"), ("sour", "เปรี้ยว"), ("sweet", "หวาน"), ("rich", "เข้มข้น"), ("fresh", "สดชื่น")]
OCCASIONS = [("lunch", "มื้อกลางวัน"), ("dinner", "มื้อเย็น"), ("group dining", "กินกับเพื่อน"), ("quick meal", "มื้อเร็ว"), ("family meal", "กินกับครอบครัว")]

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u0E00-\u0E7F]+|[^\s]")

def approx_tokens(text: str) -> int:
    # Rough estimator for mixed Thai/English without requiring BGE tokenizer.
    # Thai continuous spans are counted by chars / 3.2; English by words.
    total = 0
    for m in _WORD_RE.finditer(text):
        s = m.group(0)
        if re.fullmatch(r"[\u0E00-\u0E7F]+", s):
            total += max(1, math.ceil(len(s) / 3.2))
        elif re.fullmatch(r"[A-Za-z0-9_]+", s):
            total += 1
        else:
            total += 1
    return total

def parse_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        v = json.loads(value)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [value]

def cuisine_for(row: pd.Series) -> tuple[str, str, list[str]]:
    ids = parse_list(row.get("type_ids"))
    for type_id in ids:
        if type_id in CUISINE_MAP:
            return CUISINE_MAP[type_id]
    types = " ".join(parse_list(row.get("types"))).lower()
    if "pizza" in types:
        return CUISINE_MAP["pizza_restaurant"]
    if "cafe" in types or "coffee" in types:
        return CUISINE_MAP["cafe"]
    return CUISINE_MAP["restaurant"]

def choose_doc(row: pd.Series, bucket: str, rng: random.Random, idx: int) -> str:
    title = str(row["title"]).strip()
    rating = row.get("rating")
    reviews = row.get("reviews")
    rating_txt = f"rating {rating:.1f} from {int(reviews or 0)} reviews" if pd.notna(rating) else "rating not available"
    cuisine_en, cuisine_th, menus = cuisine_for(row)
    area_en, area_th = rng.choice(AREAS)
    taste_en, taste_th = rng.choice(TASTES)
    occasion_en, occasion_th = rng.choice(OCCASIONS)
    menu_a, menu_b, menu_c = rng.sample(menus, k=3)
    h3r8 = str(row.get("h3_r8", ""))
    lat, lon = float(row.get("latitude", 0)), float(row.get("longitude", 0))

    query_th = f"ร้าน{cuisine_th}แถว{area_th} มี{menu_a}"
    query_en = f"{cuisine_en} restaurant near {area_en} with {menu_a}"

    if bucket == "t32":
        return " ".join((
            f"{title}. {query_en}. ค้นหา: {query_th}. "
            f"Menus: {menu_a}, {menu_b}. Category: {cuisine_en}/{cuisine_th}. {rating_txt}."
        ).split())

    if bucket == "t64":
        return " ".join((
            f"Restaurant: {title}. Query TH: {query_th}. Query EN: {query_en}. "
            f"This Bangkok restaurant benchmark text tests Thai English retrieval for cuisine, menu, nearby place, and rating intent. "
            f"Popular menus include {menu_a}, {menu_b}, and {menu_c}. Taste: {taste_en}/{taste_th}. {rating_txt}."
        ).split())

    base = (
        f"Restaurant benchmark record {idx}. Query TH: {query_th} สำหรับ{occasion_th}. Query EN: {query_en} for {occasion_en}. "
        f"Document: {title} is a {cuisine_en} restaurant candidate in the Bangkok restaurant corpus. "
        f"It is useful for retrieval tests about cuisine, menu category, location intent, rating filters, and bilingual Thai English search. "
        f"Popular menus include {menu_a}, {menu_b}, and {menu_c}. Thai category: {cuisine_th}. "
        f"Taste profile: {taste_en} / {taste_th}. Occasion: {occasion_en} / {occasion_th}. "
        f"Metadata: {rating_txt}; place_id={row.get('place_id')}; h3_r8={h3r8}; lat={lat:.5f}; lon={lon:.5f}."
    )
    filler_sentences = [
        f"Thai users may search '{query_th}' while English menu data says {menu_a}, {menu_b}, or {menu_c}.",
        f"This line tests BGE-M3 tokenization and embedding for mixed scripts, restaurant names, numeric ratings, H3 cells, and menu synonyms.",
        f"Relevant categories are restaurant, menu item, cuisine category, place search, nearby dining, and intent to menu retrieval.",
        f"Hard-negative style comparisons include same area but wrong cuisine, same cuisine but wrong area, and similar menu with different dietary preference.",
        f"คำอธิบายภาษาไทย: {title} เหมาะกับการค้นหาร้าน{cuisine_th} เมนูเด่นคือ {menu_a}, {menu_b}, {menu_c} และเหมาะสำหรับ{occasion_th}.",
        f"Benchmark focus: average sequence length, batch padding cost, throughput per input, tokenize per second, embedding per second, and end-to-end latency.",
        f"Use this as one raw input line for bge-m3-bench --texts; the benchmark client sends batches over gRPC and records request metrics.",
    ]
    target = {"t128": 128, "t256": 256, "t512": 512}[bucket]
    text = base
    j = 0
    while approx_tokens(text) < target - 8:
        text += " " + filler_sentences[j % len(filler_sentences)]
        j += 1
    return " ".join(text.split())

def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.replace("\n", " ").strip() + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-csv", default="/mnt/data/seed_restaurants.csv")
    ap.add_argument("--out-dir", default="/mnt/data/bge_m3_bench_inputs")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    rng = random.Random(SEED)
    df = pd.read_csv(args.seed_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    buckets = ["t32", "t64", "t128", "t256", "t512"]
    manifest = {"source_csv": args.seed_csv, "seed_rows": int(len(df)), "generator_seed": SEED, "files": []}
    all_mixed: list[str] = []

    for bucket in buckets:
        lines = []
        for i in range(args.n):
            row = df.iloc[(i * 37 + rng.randrange(len(df))) % len(df)]
            lines.append(choose_doc(row, bucket, rng, i))
        path = out / f"restaurant_{bucket}_{args.n}.txt"
        write_lines(path, lines)
        toks = [approx_tokens(x) for x in lines]
        manifest["files"].append({
            "file": str(path), "bucket": bucket, "rows": len(lines),
            "approx_token_avg": round(sum(toks)/len(toks), 2),
            "approx_token_min": min(toks), "approx_token_max": max(toks),
            "usage": f"uv run bge-m3-bench --address localhost:50051 --texts {path} --batch-size 16 --concurrency 4 --warmup-sec 10 --duration-sec 60 --out results/{bucket}.jsonl",
        })
        all_mixed.extend(lines[:200])

    mixed_path = out / "restaurant_mixed_lengths_1000.txt"
    write_lines(mixed_path, all_mixed)
    toks = [approx_tokens(x) for x in all_mixed]
    manifest["files"].append({
        "file": str(mixed_path), "bucket": "mixed", "rows": len(all_mixed),
        "approx_token_avg": round(sum(toks)/len(toks), 2),
        "approx_token_min": min(toks), "approx_token_max": max(toks),
        "usage": f"uv run bge-m3-bench --address localhost:50051 --texts {mixed_path} --batch-size 16 --concurrency 4 --warmup-sec 10 --duration-sec 60 --out results/mixed_lengths.jsonl",
    })

    # Small smoke-test file for quick local validation.
    smoke = []
    for bucket in buckets:
        smoke.extend((out / f"restaurant_{bucket}_{args.n}.txt").read_text(encoding="utf-8").splitlines()[:4])
    smoke_path = out / "restaurant_smoke_20.txt"
    write_lines(smoke_path, smoke)
    toks = [approx_tokens(x) for x in smoke]
    manifest["files"].append({
        "file": str(smoke_path), "bucket": "smoke", "rows": len(smoke),
        "approx_token_avg": round(sum(toks)/len(toks), 2),
        "approx_token_min": min(toks), "approx_token_max": max(toks),
        "usage": f"uv run bge-m3-bench --address localhost:50051 --texts {smoke_path} --batch-size 4 --concurrency 1 --warmup-sec 1 --duration-sec 5 --out results/smoke.jsonl",
    })

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
