#!/usr/bin/env python3
"""
Build a SignalCheck eval dataset from HC3 (Human ChatGPT Comparison Corpus).

Why HC3: the human answers are real Reddit / StackExchange / expert answers
collected before ChatGPT was public, so the "human" label is true by
construction. That is the property the previous dataset lacked.

Why the preprocessing below is not optional
-------------------------------------------
Raw HC3 has two artifacts that let a classifier win for the wrong reason:

  1. LENGTH. Human answers median ~374 chars, ChatGPT ~1047. A classifier that
     only counts characters scores ~84% on raw HC3. Any "accuracy" measured on
     the raw set is mostly length detection.

  2. DETOKENIZATION. The human half is Reddit-exported with spaces before
     punctuation ('categories of " Best Seller " . Replace'). The AI half has
     normal punctuation. A detector could learn the whitespace, not the writing.

This script normalizes whitespace on BOTH halves and then length-matches them
bin by bin, so the two classes have overlapping length distributions. It ends
by reporting what a length-only classifier would score on the output: if that
number is near 50%, the confound is controlled and any signal your model finds
has to come from the text itself.

Usage:
    python build_hc3_dataset.py --inspect
    python build_hc3_dataset.py --n 100
"""
import argparse
import csv
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
CACHE = Path("hc3_all.jsonl")
HUMAN_KEYS = ("human_answers", "human_answer", "human")
AI_KEYS = ("chatgpt_answers", "chatgpt_answer", "chatgpt", "ai_answers")


def download():
    if CACHE.exists():
        print(f"Using cached {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")
        return
    print(f"Downloading HC3 -> {CACHE}")
    req = urllib.request.Request(HC3_URL, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as r, open(CACHE, "wb") as f:
        f.write(r.read())
    print(f"Done ({CACHE.stat().st_size/1e6:.1f} MB)")


def load_rows():
    with open(CACHE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pick_key(row, candidates, kind):
    for k in candidates:
        if k in row:
            return k
    sys.exit(f"\nNo {kind} field. Row keys: {list(row.keys())}\n"
             f"Edit HUMAN_KEYS / AI_KEYS at the top of this script.")


# Some HC3 rows captured the ChatGPT web UI instead of the answer -- version
# footers, "New chat" buttons, the research-preview banner. Those texts name
# their own label, so any model scores 100% on them for the wrong reason.
CONTAMINATION = re.compile(
    r"free research preview|chatgpt (?:dec|jan|feb|mar|may) \d|"
    r"new chat\b|regenerate response|our goal is to make ai systems|"
    r"as an ai language model, i (?:do not have access to|cannot browse)",
    re.I)


def is_contaminated(t: str) -> bool:
    return bool(CONTAMINATION.search(t))


def normalize(t: str) -> str:
    """
    Undo export artifacts so formatting can't stand in for authorship.
    Applied identically to both classes -- never to one side only.
    """
    t = re.sub(r"[​-‏﻿ ]", " ", t)   # zero-width / nbsp
    t = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', t)          # '" x "' -> '"x"'
    t = re.sub(r"\s+([,.!?;:%])", r"\1", t)              # ' .'  -> '.'
    t = re.sub(r'\s+([\)\]\}])', r"\1", t)               # ' )'  -> ')'
    t = re.sub(r'([\(\[\{])\s+', r"\1", t)               # '( '  -> '('
    t = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", t)          # 'oscar - winning'
    t = re.sub(r"\s+('[sdtm]\b|'re\b|'ve\b|'ll\b|n't\b)", r"\1", t)
    # Collapse ALL whitespace including newlines. 39% of raw AI texts contained
    # line breaks vs 3% of human ones, so layout alone leaked the label.
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def length_only_accuracy(samples):
    """Best accuracy achievable by thresholding on character count alone."""
    h = sorted(len(s["text_content"]) for s in samples if s["label"] == "human")
    a = sorted(len(s["text_content"]) for s in samples if s["label"] == "ai")
    if not h or not a:
        return 0.0
    lo, hi = min(h[0], a[0]), max(h[-1], a[-1])
    best = 0.0
    for t in range(lo, hi + 1, max(1, (hi - lo) // 400)):
        # try both directions; a confound in either is still a confound
        f = (sum(1 for x in h if x < t) + sum(1 for x in a if x >= t)) / (len(h) + len(a))
        best = max(best, f, 1 - f)
    return best * 100


def collect(rows, key, label, min_chars, max_chars):
    out = []
    for i, r in enumerate(rows):
        v = r.get(key) or []
        for j, t in enumerate(v if isinstance(v, list) else [v]):
            t = normalize(t or "")
            if is_contaminated(t):
                continue
            if min_chars <= len(t) <= max_chars:
                out.append({"text_id": f"{label}_{i}_{j}", "label": label,
                            "text_content": t, "src": r.get("source", "")})
    return out


def length_match(human, ai, k_per_class, bin_size, rng):
    """
    Sample so both classes have the same length histogram: bin by character
    count, and from every bin take an equal number from each class.
    """
    hb, ab = defaultdict(list), defaultdict(list)
    for s in human:
        hb[len(s["text_content"]) // bin_size].append(s)
    for s in ai:
        ab[len(s["text_content"]) // bin_size].append(s)

    shared = sorted(set(hb) & set(ab))
    capacity = {b: min(len(hb[b]), len(ab[b])) for b in shared}
    total = sum(capacity.values())
    if total == 0:
        sys.exit("No overlapping length bins -- widen --min/--max-chars.")

    picked_h, picked_a = [], []
    for b in shared:                       # proportional draw from each bin
        take = min(capacity[b], max(1, round(k_per_class * capacity[b] / total)))
        picked_h += rng.sample(hb[b], take)
        picked_a += rng.sample(ab[b], take)

    k = min(k_per_class, len(picked_h), len(picked_a))
    return rng.sample(picked_h, k), rng.sample(picked_a, k)


def inspect():
    rows = load_rows()
    hk, ak = pick_key(rows[0], HUMAN_KEYS, "human"), pick_key(rows[0], AI_KEYS, "AI")
    print(f"\nrows: {len(rows)}   keys: {list(rows[0].keys())}")
    print(f"human field: {hk!r}   ai field: {ak!r}\n")
    raw = rows[0][hk][0] if isinstance(rows[0][hk], list) else rows[0][hk]
    print("--- HUMAN raw ---\n", raw[:220].strip())
    print("\n--- HUMAN normalized ---\n", normalize(raw)[:220])
    for kind, key in (("human", hk), ("ai", ak)):
        L = sorted(len(normalize(t)) for r in rows[:3000]
                   for t in (r.get(key) or []) if t)
        print(f"\n{kind:>5}: median {L[len(L)//2]}  p10 {L[len(L)//10]}  p90 {L[9*len(L)//10]}")


def build(k, min_chars, max_chars, bin_size, seed, out_path):
    rows = load_rows()
    hk, ak = pick_key(rows[0], HUMAN_KEYS, "human"), pick_key(rows[0], AI_KEYS, "AI")
    rng = random.Random(seed)

    human = collect(rows, hk, "human", min_chars, max_chars)
    ai = collect(rows, ak, "ai", min_chars, max_chars)
    print(f"eligible after normalize + {min_chars}-{max_chars} chars: "
          f"human {len(human)}  ai {len(ai)}")

    before = length_only_accuracy(rng.sample(human, min(400, len(human)))
                                  + rng.sample(ai, min(400, len(ai))))
    ph, pa = length_match(human, ai, k, bin_size, rng)
    sample = ph + pa
    rng.shuffle(sample)
    after = length_only_accuracy(sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text_id", "text_content", "label"])
        w.writeheader()
        w.writerows([{kk: s[kk] for kk in ("text_id", "text_content", "label")}
                     for s in sample])

    chk = list(csv.DictReader(open(out_path, encoding="utf-8")))
    assert {r["label"] for r in chk} == {"ai", "human"}
    nh = sum(1 for r in chk if r["label"] == "human")
    L = sorted(len(r["text_content"]) for r in chk)
    Lh = sorted(len(r["text_content"]) for r in chk if r["label"] == "human")
    La = sorted(len(r["text_content"]) for r in chk if r["label"] == "ai")

    print(f"\nwrote {out_path}  ({len(chk)} rows)")
    print(f"  balance:        {nh} human / {len(chk)-nh} ai")
    print(f"  length median:  human {Lh[len(Lh)//2]}  ai {La[len(La)//2]}   "
          f"(overall {L[len(L)//2]}, range {L[0]}-{L[-1]})")
    print(f"\n  AUDIT — class-specific formatting artifacts:")
    probes = {
        "space-hyphen ' - '": lambda s: " - " in s,
        "newline":            lambda s: "\n" in s,
        "quote spacing":      lambda s: bool(re.search(r'\s"\s', s)),
        "'chatgpt' in text":  lambda s: "chatgpt" in s.lower(),
    }
    worst = 0
    for nm, fn in probes.items():
        ph = 100*sum(fn(r["text_content"]) for r in chk if r["label"]=="human")/max(nh,1)
        pa = 100*sum(fn(r["text_content"]) for r in chk if r["label"]=="ai")/max(len(chk)-nh,1)
        worst = max(worst, abs(ph-pa))
        mark = "  <-- LEAKAGE" if abs(ph-pa) > 15 else ""
        print(f"    {nm:<20} human {ph:>4.0f}%   ai {pa:>4.0f}%   gap {abs(ph-pa):>4.0f}%{mark}")
    print(f"    {'worst gap':<20} {worst:.0f}%   {'OK' if worst<=15 else 'FIX BEFORE USING'}")

    print(f"\n  AUDIT — accuracy from character count alone:")
    print(f"    before length matching: {before:.0f}%")
    print(f"    in this file:           {after:.0f}%   "
          f"{'OK' if after < 60 else 'STILL CONFOUNDED — lower --bin-size'}")
    print(f"  majority-class baseline:  {max(nh, len(chk)-nh)/len(chk)*100:.0f}%")
    print(f"  -> your model must beat both numbers to have measured anything.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--n", type=int, default=100, help="samples per class")
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-chars", type=int, default=2000, help="production truncation")
    p.add_argument("--bin-size", type=int, default=100, help="length-matching bin width")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="datasets/hc3_eval.csv")
    a = p.parse_args()
    download()
    inspect() if a.inspect else build(a.n, a.min_chars, a.max_chars,
                                      a.bin_size, a.seed, Path(a.out))
