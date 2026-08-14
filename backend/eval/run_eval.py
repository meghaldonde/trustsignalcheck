#!/usr/bin/env python3
"""
SignalCheck V3 Eval Harness

Usage:
    python -m eval.run_eval scan [--sample N] [--dataset PATH]
    python -m eval.run_eval score [--threshold T]

The harness calls analyze_with_gemini() directly, bypassing:
- /api/scan endpoint
- Rate limiter
- Safe Browsing API
- Scans DB (24h purge would delete predictions)
- Dashboard aggregates (would pollute unit economics)

Predictions are stored in eval/predictions/{prompt_version}.jsonl
keyed by (prompt_hash, text_id). Changing the prompt invalidates the cache.
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai
from pydantic import ValidationError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import AIAnalysis
from eval.config import (
    PROMPT_VERSION,
    PROMPT_HASH,
    PROMPT_TEMPLATE,
    MODEL_NAME,
    TEXT_TRUNCATION,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_DELAY_SECONDS,
    GEMINI_INPUT_COST_PER_1M,
    GEMINI_OUTPUT_COST_PER_1M,
    DEFAULT_DATASET_PATH,
    DEFAULT_SAMPLE_SIZE,
    VARIANCE_TEXTS,
    VARIANCE_TRIALS,
)


@dataclass
class PredictionRow:
    """A single prediction result."""
    text_id: str
    prompt_version: str
    prompt_hash: str
    label: str  # ground truth from dataset
    ai_score: Optional[int]  # predicted score, None if parse failure
    status: str  # "success" or "parse_failure"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    reasoning: Optional[str]
    key_signals: Optional[list]
    raw_response: Optional[str]  # stored on parse failure for debugging
    timestamp: str


def get_predictions_path() -> Path:
    """Get path to predictions file for current prompt version."""
    return Path(__file__).parent / "predictions" / f"{PROMPT_VERSION}.jsonl"


def load_existing_predictions() -> dict[str, PredictionRow]:
    """
    Load existing predictions for resume functionality.
    Returns dict keyed by text_id.
    Validates that all rows match current prompt_hash.
    """
    path = get_predictions_path()
    if not path.exists():
        return {}

    predictions = {}
    mismatched_hash = None

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            # Check hash matches
            if row.get("prompt_hash") != PROMPT_HASH:
                mismatched_hash = row.get("prompt_hash")
                break

            # API errors (429s, timeouts) are transport failures, not results.
            # Skipping them here means the next run retries those text_ids
            # instead of treating them as already processed.
            if row.get("status") == "api_error":
                continue

            predictions[row["text_id"]] = row

    if mismatched_hash:
        print(f"\n{'='*60}")
        print(f"HASH MISMATCH")
        print(f"{'='*60}")
        print(f"File hash:    {mismatched_hash}")
        print(f"Current hash: {PROMPT_HASH}")
        print(f"\nThe prompt has changed but PROMPT_VERSION is still '{PROMPT_VERSION}'.")
        print(f"Bump PROMPT_VERSION in eval/config.py to start a new predictions file.")
        print(f"{'='*60}\n")
        sys.exit(1)

    return predictions


def append_prediction(row: dict):
    """Append a single prediction to the JSONL file (crash-safe)."""
    path = get_predictions_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_dataset(dataset_path: str, sample_size: Optional[int] = None) -> list[dict]:
    """Load dataset from CSV with validation."""
    path = Path(__file__).parent / dataset_path
    if not path.exists():
        # Try absolute path
        path = Path(dataset_path)

    if not path.exists():
        print(f"Dataset not found: {path}")
        sys.exit(1)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns exist
        required_columns = {"text_id", "text_content", "label"}
        if reader.fieldnames is None:
            print(f"ERROR: CSV has no headers")
            sys.exit(1)
        missing = required_columns - set(reader.fieldnames)
        if missing:
            print(f"ERROR: Missing required columns: {missing}")
            print(f"Found columns: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            rows.append(row)

    # Validate labels are expected values.
    # 'unknown' is allowed for datasets that measure score MOVEMENT rather than
    # accuracy -- e.g. the metadata A/B, where the same prose is scanned with and
    # without a byline and there is no ground truth to be right or wrong about.
    # Those rows are excluded from accuracy in run_score().
    labels = {r["label"] for r in rows}
    if not labels <= {"ai", "human", "unknown"}:
        print(f"ERROR: Unexpected labels {labels}; expected 'ai'/'human'/'unknown'")
        print(f"Map them in load_dataset() before proceeding.")
        sys.exit(1)
    if "unknown" in labels:
        n_unknown = sum(1 for r in rows if r["label"] == "unknown")
        print(f"NOTE: {n_unknown} row(s) labelled 'unknown' -- scored for "
              f"movement, excluded from accuracy.")

    if sample_size and sample_size < len(rows):
        import random
        random.seed(42)  # Reproducible sampling
        rows = random.sample(rows, sample_size)

    return rows


# Gemini client (lazy init)
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("GOOGLE_API_KEY environment variable must be set")
            sys.exit(1)
        _client = genai.Client(api_key=api_key)
    return _client


async def analyze_text(text: str, semaphore: asyncio.Semaphore) -> dict:
    """
    Analyze a single text sample with Gemini.
    Returns dict with prediction data.
    """
    async with semaphore:
        prompt = PROMPT_TEMPLATE.format(text=text[:TEXT_TRUNCATION])

        try:
            client = get_client()

            # Run sync API call in executor
            loop = asyncio.get_running_loop()
            interaction = await loop.run_in_executor(
                None,
                lambda: client.interactions.create(
                    model=MODEL_NAME,
                    input=prompt,
                )
            )

            response_text = interaction.output_text.strip()

            # Strip markdown fences
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            # Get token counts
            usage = getattr(interaction, "usage", None)
            if usage:
                input_tokens = getattr(usage, "total_input_tokens", 0) or 0
                output_tokens = (getattr(usage, "total_output_tokens", 0) or 0) + \
                               (getattr(usage, "total_thought_tokens", 0) or 0)
            else:
                input_tokens = len(prompt) // 4
                output_tokens = len(response_text) // 4

            cost_usd = (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_1M + \
                       (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M

            # Parse response
            try:
                analysis = AIAnalysis.model_validate_json(response_text)
                return {
                    "status": "success",
                    "ai_score": analysis.ai_probability_score,
                    "reasoning": analysis.reasoning_flag,
                    "key_signals": analysis.key_signals,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "raw_response": None,
                }
            except (ValidationError, json.JSONDecodeError) as e:
                # Schema compliance failure - record it
                return {
                    "status": "parse_failure",
                    "ai_score": None,
                    "reasoning": None,
                    "key_signals": None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "raw_response": response_text[:500],  # Truncate for storage
                }

        except Exception as e:
            # API error - still record for debugging
            return {
                "status": "api_error",
                "ai_score": None,
                "reasoning": None,
                "key_signals": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "raw_response": str(e)[:500],
            }

        finally:
            # Small delay to avoid hitting rate limits
            await asyncio.sleep(REQUEST_DELAY_SECONDS)


def estimate_cost_per_scan() -> float:
    """Derive cost estimate from pricing constants (not hardcoded)."""
    # ~500 input tokens, ~700 output tokens per scan (observed average)
    avg_input = 500
    avg_output = 700
    return (avg_input / 1_000_000) * GEMINI_INPUT_COST_PER_1M + \
           (avg_output / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M


async def process_single(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """Process a single row and return the prediction dict."""
    text_id = row["text_id"]
    label = row["label"]
    text = row["text_content"]

    result = await analyze_text(text, semaphore)

    return {
        "text_id": text_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "label": label,
        "ai_score": result["ai_score"],
        "status": result["status"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
        "reasoning": result["reasoning"],
        "key_signals": result["key_signals"],
        "raw_response": result["raw_response"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def run_scan(dataset_path: str, sample_size: Optional[int] = None):
    """Run scan command - generate predictions for dataset."""
    print(f"\n{'='*60}")
    print(f"SignalCheck Eval Harness - SCAN")
    print(f"{'='*60}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Prompt hash:    {PROMPT_HASH}")
    print(f"Model:          {MODEL_NAME}")
    print(f"Concurrency:    {MAX_CONCURRENT_REQUESTS}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset(dataset_path, sample_size)
    print(f"Dataset: {len(dataset)} samples")

    # Load existing predictions for resume
    existing = load_existing_predictions()
    print(f"Existing predictions: {len(existing)} (will skip)")

    # Filter to remaining samples
    remaining = [row for row in dataset if row["text_id"] not in existing]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("\nAll samples already processed!")
        return

    # Estimate cost and time (using derived cost, not hardcoded)
    cost_per_scan = estimate_cost_per_scan()
    est_cost = len(remaining) * cost_per_scan
    # ~6s per call, divided by concurrency
    est_time_mins = (len(remaining) * 6 / MAX_CONCURRENT_REQUESTS) / 60
    print(f"\nEstimated cost: ${est_cost:.2f}")
    print(f"Estimated time: {est_time_mins:.1f} minutes (with {MAX_CONCURRENT_REQUESTS} concurrent)")
    print(f"\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")

    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Process in concurrent chunks using asyncio.gather
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    start_time = time.time()

    success_count = 0
    parse_failure_count = 0
    api_error_count = 0
    total_cost = 0.0
    processed = 0

    # Process in chunks to allow progress updates while maintaining concurrency
    chunk_size = MAX_CONCURRENT_REQUESTS * 2  # Process 2 "waves" at a time

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start:chunk_start + chunk_size]

        # Launch all tasks in chunk concurrently
        tasks = [process_single(row, semaphore) for row in chunk]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and append immediately (crash-safe)
        for prediction in results:
            if isinstance(prediction, Exception):
                # Task raised an exception
                api_error_count += 1
                continue

            append_prediction(prediction)
            processed += 1

            total_cost += prediction["cost_usd"]
            if prediction["status"] == "success":
                success_count += 1
            elif prediction["status"] == "parse_failure":
                parse_failure_count += 1
            else:
                api_error_count += 1

        # Progress update after each chunk
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - processed) / rate if rate > 0 else 0

        print(f"\r[{processed}/{len(remaining)}] "
              f"Success: {success_count} | "
              f"Parse fail: {parse_failure_count} | "
              f"API error: {api_error_count} | "
              f"Cost: ${total_cost:.4f} | "
              f"ETA: {eta/60:.1f}m", end="", flush=True)

    print(f"\n\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed:    {processed}")
    print(f"Success:            {success_count}")
    print(f"Parse failures:     {parse_failure_count}")
    print(f"API errors:         {api_error_count}")
    print(f"Total cost:         ${total_cost:.4f}")
    # Schema compliance: valid / (valid + parse_failures), not including API errors
    schema_denom = success_count + parse_failure_count
    schema_compliance = 100 * success_count / schema_denom if schema_denom > 0 else 0
    print(f"Schema compliance:  {schema_compliance:.1f}% ({success_count}/{schema_denom})")
    print(f"Predictions file:   {get_predictions_path()}")
    print(f"{'='*60}\n")


def get_variance_path() -> Path:
    """Get path to variance results file for current prompt version."""
    return Path(__file__).parent / "predictions" / f"variance-{PROMPT_VERSION}.jsonl"


async def run_variance(dataset_path: str, num_texts: int, num_trials: int):
    """
    Run variance check - same texts multiple times to measure score stability.
    If identical input yields materially different scores, single-run accuracy is noise.
    """
    print(f"\n{'='*60}")
    print(f"SignalCheck Eval Harness - VARIANCE CHECK")
    print(f"{'='*60}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Prompt hash:    {PROMPT_HASH}")
    print(f"Texts:          {num_texts}")
    print(f"Trials/text:    {num_trials}")
    print(f"Total API calls: {num_texts * num_trials}")
    print(f"{'='*60}\n")

    # Load dataset (small sample)
    dataset = load_dataset(dataset_path, num_texts)
    print(f"Selected {len(dataset)} texts for variance testing")

    # Estimate cost (derived, not hardcoded)
    cost_per_scan = estimate_cost_per_scan()
    est_cost = len(dataset) * num_trials * cost_per_scan
    est_time_mins = (len(dataset) * num_trials * 6 / MAX_CONCURRENT_REQUESTS) / 60
    print(f"Estimated cost: ${est_cost:.2f}")
    print(f"Estimated time: {est_time_mins:.1f} minutes")
    print(f"\nPress Ctrl+C to cancel, or wait 3 seconds to continue...")

    try:
        await asyncio.sleep(3)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = {}  # text_id -> {label, scores, text}

    # Build all tasks upfront for true concurrency
    all_tasks = []
    for row in dataset:
        text_id = row["text_id"]
        text = row["text_content"]
        label = row["label"]
        results[text_id] = {"label": label, "text": text, "scores": [], "raw_results": []}

        for trial in range(num_trials):
            all_tasks.append((text_id, text, trial))

    total_calls = len(all_tasks)
    start_time = time.time()

    # Process in concurrent chunks
    chunk_size = MAX_CONCURRENT_REQUESTS * 2
    completed = 0

    for chunk_start in range(0, len(all_tasks), chunk_size):
        chunk = all_tasks[chunk_start:chunk_start + chunk_size]

        async def process_variance_task(text_id, text, trial):
            result = await analyze_text(text, semaphore)
            return text_id, trial, result

        tasks = [process_variance_task(tid, txt, tr) for tid, txt, tr in chunk]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in chunk_results:
            if isinstance(item, Exception):
                completed += 1
                continue

            text_id, trial, result = item
            if result["status"] == "success":
                results[text_id]["scores"].append(result["ai_score"])
            else:
                results[text_id]["scores"].append(None)
            results[text_id]["raw_results"].append(result)
            completed += 1

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total_calls - completed) / rate if rate > 0 else 0
        print(f"\r[{completed}/{total_calls}] ETA: {eta:.0f}s", end="", flush=True)

    print(f"\n\n{'='*60}")
    print(f"VARIANCE RESULTS")
    print(f"{'='*60}\n")

    # Compute stats
    import statistics

    print(f"{'Text ID':<12} {'Label':<8} {'Scores':<30} {'Mean':>8} {'Stdev':>8}")
    print(f"{'-'*76}")

    variance_data = []
    stdevs = []

    for text_id, data in results.items():
        scores = [s for s in data["scores"] if s is not None]
        row_data = {
            "text_id": text_id,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": PROMPT_HASH,
            "label": data["label"],
            "scores": scores,
            "num_trials": num_trials,
            "num_success": len(scores),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if len(scores) >= 2:
            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores)
            stdevs.append(stdev)
            row_data["mean"] = mean
            row_data["stdev"] = stdev
            scores_str = ", ".join(str(s) for s in scores)
            print(f"{text_id:<12} {data['label']:<8} {scores_str:<30} {mean:>8.1f} {stdev:>8.1f}")
        elif scores:
            row_data["mean"] = scores[0]
            row_data["stdev"] = None
            print(f"{text_id:<12} {data['label']:<8} {scores[0]:<30} {'--':>8} {'--':>8}")
        else:
            row_data["mean"] = None
            row_data["stdev"] = None
            print(f"{text_id:<12} {data['label']:<8} {'(all failed)':<30} {'--':>8} {'--':>8}")

        variance_data.append(row_data)

    # Summary stats
    summary = {
        "type": "summary",
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "num_texts": num_texts,
        "num_trials": num_trials,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    verdict = None
    print(f"\n{'='*60}")
    if stdevs:
        avg_stdev = statistics.mean(stdevs)
        max_stdev = max(stdevs)
        summary["avg_stdev"] = avg_stdev
        summary["max_stdev"] = max_stdev

        print(f"Average stdev:  {avg_stdev:.1f}")
        print(f"Max stdev:      {max_stdev:.1f}")
        print(f"")
        if avg_stdev < 5:
            verdict = "low_variance"
            print(f"VERDICT: Low variance - single-pass accuracy is defensible")
        elif avg_stdev < 10:
            verdict = "moderate_variance"
            print(f"VERDICT: Moderate variance - consider multiple runs for stability")
        else:
            verdict = "high_variance"
            print(f"VERDICT: High variance - single-pass accuracy includes significant noise")
            print(f"         This is a product finding: trust scores are not stable on identical input")
        summary["verdict"] = verdict
    else:
        summary["verdict"] = "insufficient_data"
        print(f"Not enough successful trials to compute variance")

    # Write results to disk
    variance_path = get_variance_path()
    variance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variance_path, "w") as f:
        f.write(json.dumps(summary) + "\n")
        for row in variance_data:
            f.write(json.dumps(row) + "\n")

    print(f"\nResults saved to: {variance_path}")
    print(f"{'='*60}\n")


def run_pairs():
    """
    Compare paired variants of the same prose.

    Built for the metadata A/B: rows named "<page>__with_metadata" and
    "<page>__prose_only" carry identical text apart from a byline / date /
    reading-time block. Any score difference is attributable to that block,
    which is the point -- it measures whether the model is judging provenance
    rather than writing.
    """
    predictions = load_existing_predictions()
    pairs = {}
    for p in predictions.values():
        if p["status"] != "success" or "__" not in p["text_id"]:
            continue
        stem, variant = p["text_id"].rsplit("__", 1)
        pairs.setdefault(stem, {})[variant] = p

    complete = {k: v for k, v in pairs.items() if len(v) >= 2}
    if not complete:
        print("No complete pairs found. Expected ids like '<page>__with_metadata' "
              "and '<page>__prose_only'.")
        return

    print(f"\n{'='*72}")
    print("PAIRED COMPARISON — identical prose, metadata block varied")
    print(f"{'='*72}\n")
    print(f"{'page':<32}{'with meta':>11}{'prose only':>12}{'delta':>8}")
    print("-" * 72)

    deltas = []
    for stem, v in sorted(complete.items()):
        a = v.get("with_metadata", {}).get("ai_score")
        b = v.get("prose_only", {}).get("ai_score")
        if a is None or b is None:
            continue
        d = b - a                      # positive => metadata LOWERED the AI score
        deltas.append(d)
        print(f"{stem[:31]:<32}{a:>10}%{b:>11}%{d:>+7}")

    if deltas:
        mean = sum(deltas) / len(deltas)
        print("-" * 72)
        print(f"{'mean delta':<32}{'':>10}{'':>11}{mean:>+7.1f}")
        print()
        if mean > 10:
            print("Metadata LOWERS the AI score. The model is weighting provenance")
            print("(byline, date, reading time) over the writing itself, which means")
            print("generated text can reduce its own score by adding a fake byline.")
        elif mean < -10:
            print("Metadata RAISES the AI score -- the opposite of the hypothesis.")
        else:
            print("No meaningful movement. Provenance cues in the metadata block are")
            print("not driving the score; the reasoning text was narrating, not deciding.")
        print("\nNote: n is small. Treat this as directional unless the pattern")
        print("holds across more pages.")
    print()


def run_score(threshold: int = 50):
    """Run score command - compute metrics from predictions."""
    print(f"\n{'='*60}")
    print(f"SignalCheck Eval Harness - SCORE")
    print(f"{'='*60}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Prompt hash:    {PROMPT_HASH}")
    print(f"Threshold:      {threshold}")
    print(f"{'='*60}\n")

    # Load predictions
    predictions = load_existing_predictions()
    if not predictions:
        print("No predictions found. Run 'scan' first.")
        return

    print(f"Loaded {len(predictions)} predictions")

    # Filter to successful predictions only
    valid = [p for p in predictions.values() if p["status"] == "success"]
    parse_failures = [p for p in predictions.values() if p["status"] == "parse_failure"]
    api_errors = [p for p in predictions.values() if p["status"] == "api_error"]

    # Rows with no ground truth cannot be right or wrong. Counting them as
    # 'human' (which `label == "ai"` would do silently) would corrupt accuracy.
    unlabelled = [p for p in valid if p.get("label") not in ("ai", "human")]
    valid = [p for p in valid if p.get("label") in ("ai", "human")]

    print(f"Valid predictions: {len(valid)}")
    print(f"Parse failures:    {len(parse_failures)}")
    print(f"API errors:        {len(api_errors)}")
    if unlabelled:
        print(f"Unlabelled:        {len(unlabelled)} (excluded from accuracy; "
              f"use 'pairs' to compare them)")

    if not valid:
        print("No valid predictions to score.")
        return

    # Compute metrics at given threshold
    # ai_score >= threshold → predict "ai"
    # ai_score < threshold → predict "human"
    true_positives = 0   # Correctly identified AI
    true_negatives = 0   # Correctly identified human
    false_positives = 0  # Human flagged as AI
    false_negatives = 0  # AI flagged as human

    for p in valid:
        predicted_ai = p["ai_score"] >= threshold
        actual_ai = p["label"] == "ai"

        if predicted_ai and actual_ai:
            true_positives += 1
        elif not predicted_ai and not actual_ai:
            true_negatives += 1
        elif predicted_ai and not actual_ai:
            false_positives += 1
        else:
            false_negatives += 1

    total = len(valid)
    accuracy = (true_positives + true_negatives) / total * 100

    # False positive rate: FP / (FP + TN) - human texts misclassified as AI
    fp_rate = false_positives / (false_positives + true_negatives) * 100 if (false_positives + true_negatives) > 0 else 0

    # False negative rate: FN / (FN + TP) - AI texts misclassified as human
    fn_rate = false_negatives / (false_negatives + true_positives) * 100 if (false_negatives + true_positives) > 0 else 0

    # Schema compliance: valid / (valid + parse_failures)
    # API errors are infrastructure failures, not schema violations
    schema_denom = len(valid) + len(parse_failures)
    schema_compliance = len(valid) / schema_denom * 100 if schema_denom > 0 else 0

    # Cost stats (from valid + parse_failures, API errors have 0 cost)
    total_cost = sum(p.get("cost_usd", 0) for p in predictions.values())
    avg_cost = total_cost / len(predictions) if predictions else 0

    print(f"\n{'='*60}")
    print(f"METRICS @ threshold={threshold}")
    print(f"{'='*60}")
    print(f"")
    print(f"  Accuracy:           {accuracy:.1f}%  (target: >80%)")
    print(f"  False Positive Rate: {fp_rate:.1f}%  (target: <5%)")
    print(f"  False Negative Rate: {fn_rate:.1f}%  (target: <20%)")
    print(f"  Schema Compliance:  {schema_compliance:.1f}%  (target: 100%) [{len(valid)}/{schema_denom}]")
    if api_errors:
        print(f"  API Errors:         {len(api_errors)} (excluded from schema compliance)")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"                      Predicted AI    Predicted Human")
    print(f"    Actual AI         {true_positives:>8}        {false_negatives:>8}")
    print(f"    Actual Human      {false_positives:>8}        {true_negatives:>8}")
    print(f"")
    print(f"  Total cost:         ${total_cost:.4f}")
    print(f"  Avg cost/sample:    ${avg_cost:.6f}")
    print(f"{'='*60}")

    # Threshold sweep
    print(f"\nThreshold Sweep:")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'FP Rate':>10} {'FN Rate':>10}")
    print(f"{'-'*42}")

    for t in [30, 40, 50, 60, 70, 80]:
        tp = sum(1 for p in valid if p["ai_score"] >= t and p["label"] == "ai")
        tn = sum(1 for p in valid if p["ai_score"] < t and p["label"] == "human")
        fp = sum(1 for p in valid if p["ai_score"] >= t and p["label"] == "human")
        fn = sum(1 for p in valid if p["ai_score"] < t and p["label"] == "ai")

        acc = (tp + tn) / total * 100
        fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

        marker = " <--" if t == threshold else ""
        print(f"{t:>10} {acc:>9.1f}% {fpr:>9.1f}% {fnr:>9.1f}%{marker}")

    print(f"\n")


def main():
    parser = argparse.ArgumentParser(
        description="SignalCheck V3 Eval Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 1. Start small - prove pipeline works (~$0.33, ~5 mins)
    python -m eval.run_eval scan

    # 2. Check variance - is the model deterministic?
    python -m eval.run_eval variance

    # 3. Score predictions and tune threshold
    python -m eval.run_eval score
    python -m eval.run_eval score --threshold 60

    # 4. Full benchmark once pipeline is proven (~$13, ~25 mins)
    python -m eval.run_eval scan --sample 2000
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Generate predictions from dataset")
    scan_parser.add_argument(
        "--sample", "-n",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Process N samples (default: {DEFAULT_SAMPLE_SIZE}, use 2000 for full run)"
    )
    scan_parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to dataset CSV (default: {DEFAULT_DATASET_PATH})"
    )

    # Variance command
    variance_parser = subparsers.add_parser("variance", help="Check score stability across trials")
    variance_parser.add_argument(
        "--texts", "-t",
        type=int,
        default=VARIANCE_TEXTS,
        help=f"Number of texts to test (default: {VARIANCE_TEXTS})"
    )
    variance_parser.add_argument(
        "--trials", "-r",
        type=int,
        default=VARIANCE_TRIALS,
        help=f"Trials per text (default: {VARIANCE_TRIALS})"
    )
    variance_parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to dataset CSV (default: {DEFAULT_DATASET_PATH})"
    )

    # Score command
    subparsers.add_parser("pairs",
        help="Compare paired variants (metadata A/B) from predictions")

    score_parser = subparsers.add_parser("score", help="Compute metrics from predictions")
    score_parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=50,
        help="AI score threshold for classification (default: 50)"
    )

    args = parser.parse_args()

    if args.command == "scan":
        asyncio.run(run_scan(args.dataset, args.sample))
    elif args.command == "variance":
        asyncio.run(run_variance(args.dataset, args.texts, args.trials))
    elif args.command == "pairs":
        run_pairs()
    elif args.command == "score":
        run_score(args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
