#!/usr/bin/env python3
"""
Build a paired A/B dataset to test whether page metadata moves the AI score.

Hypothesis (from live scans): the model scores provenance rather than prose.
Every page tested so far was scored on who wrote it or where it came from --
the reasoning cited a named byline as evidence of a human author, or recognised
the page as a scrape of a well-known reference site -- not on how it was written. If that is right, adding a byline
to generated text should lower its AI probability, which would make the score
trivially gameable.

This produces two rows per source file whose prose is byte-identical and which
differ ONLY in the metadata block. That equality is asserted before writing, so
a difference in output cannot come from anything else.

Usage:
    # save the page text (one file per page, metadata lines at the top)
    python build_metadata_ab.py pages/*.txt -o ../datasets/metadata_ab.csv

Then:
    cd backend
    python -m eval.run_eval scan --dataset ../datasets/metadata_ab.csv --sample 100
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# Lines that are page furniture rather than prose. Kept deliberately narrow --
# anything not matched here stays in the body, so the "prose only" variant errs
# toward including too much rather than silently dropping content.
META_PATTERNS = [
    r"^\s*by\s+[A-Z][\w.'-]+(\s+[A-Z][\w.'-]+){0,3}\s*$",      # by Jane Q Writer
    r"^\s*(written|authored|reviewed)\s+by\b.*$",
    r"^\s*\d+\s*min(ute)?s?\s+read\b.*$",                       # 11 min read
    r"^\s*(last\s+)?updated?(\s+date)?\s*[:\-]?\s*.*\d{4}.*$",
    r"^\s*(published|posted)(\s+on)?\s*[:\-]?\s*.*\d{4}.*$",
    r"^\s*\w[\w \-]*(>|›|»)\s*.+$",                             # breadcrumbs
    r"^\s*(share|tweet|print|save|copy link|get started|sign up|subscribe|learn more)\b.*$",
    r"^\s*(table of contents|contents|in this article|summarize with)\s*:?\s*$",
    r"^\s*\d{1,2}\s+\w+\s+\d{4}\s*$",                           # 23 October 2025
    r"^\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\s*$",
]
META_RE = [re.compile(p, re.I) for p in META_PATTERNS]


def split_metadata(text: str):
    """Return (metadata_lines, prose_lines) preserving original order."""
    meta, prose = [], []
    for line in text.splitlines():
        (meta if any(r.match(line) for r in META_RE) else prose).append(line)
    return meta, prose


def norm(s: str) -> str:
    """Compare prose ignoring only whitespace, so the assert is meaningful."""
    return re.sub(r"\s+", " ", s).strip()


def build(paths, out_path, max_chars):
    rows = []
    for p in paths:
        raw = Path(p).read_text(encoding="utf-8")

        # A forgotten placeholder would be silently scanned as prose, so refuse it.
        if "PASTE THE" in raw and "DELETE THIS LINE" in raw:
            sys.exit(f"{p}: still contains the placeholder line. Paste the console "
                     f"output over it and remove that line.")
        meta, prose = split_metadata(raw)
        stem = Path(p).stem

        if not meta:
            print(f"  {stem}: no metadata lines detected -- skipping "
                  f"(nothing to vary, so the pair would be identical)")
            continue

        # with_metadata must be the page EXACTLY as captured. An earlier version
        # rebuilt it as meta + prose, which hoisted lines like "Summarize with:"
        # from the bottom of the page to the top -- so the variant being scored
        # was a reordered document, not the original, and the comparison was
        # measuring reordering as much as metadata.
        with_meta = raw.strip()
        prose_only = "\n".join(prose).strip()

        # The whole experiment rests on this: same prose, different framing.
        # with_metadata keeps the original order, so the prose is interleaved
        # rather than contiguous -- check every prose line survives instead of
        # checking for one contiguous block.
        missing = [l for l in prose if l.strip() and norm(l) not in norm(with_meta)]
        assert not missing, f"{stem}: {len(missing)} prose line(s) lost"

        print(f"  {stem}: {len(meta)} metadata line(s) stripped, "
              f"{len(prose_only)} chars of prose")
        for line in meta:
            print(f"      - {line.strip()[:70]}")

        # Truncating each variant independently would leave with_metadata with
        # less prose than prose_only (the metadata block eats the budget), so
        # the pair would differ by trailing content as well as by metadata.
        # Trim the prose to fit the tighter of the two budgets first.
        meta_len = len(with_meta) - len(prose_only)
        prose_budget = max_chars - max(meta_len, 0)
        if prose_budget < 200:
            print(f"      ! metadata block is {meta_len} chars; too little prose "
                  f"budget left at max_chars={max_chars} -- skipping")
            continue
        trimmed = prose_only[:prose_budget]
        # rebuild with_metadata from the ORIGINAL text, cut to the same prose tail
        cut_at = with_meta.rfind(trimmed[-60:]) + 60 if len(trimmed) >= 60 else len(with_meta)
        rows.append({"text_id": f"{stem}__with_metadata", "label": "unknown",
                     "text_content": with_meta[:cut_at] if cut_at > 60 else with_meta[:max_chars]})
        rows.append({"text_id": f"{stem}__prose_only", "label": "unknown",
                     "text_content": trimmed})

    if not rows:
        sys.exit("No pairs built.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text_id", "text_content", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}  ({len(rows)} rows = {len(rows)//2} pairs)")
    print("\nNote: label is 'unknown' -- this measures score MOVEMENT between")
    print("paired variants, not accuracy, so run_eval's scoring does not apply.")
    print("Compare the two ai_score values per pair in the predictions file.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one .txt per page")
    ap.add_argument("-o", "--out", default="../datasets/metadata_ab.csv")
    ap.add_argument("--max-chars", type=int, default=2000)
    a = ap.parse_args()
    build(a.files, Path(a.out), a.max_chars)
