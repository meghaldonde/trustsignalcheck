"""
Eval harness configuration.

Prompt versioning uses both a human-readable label and a computed hash.
- Label: for file naming and readability
- Hash: for correctness (catches silent prompt edits)

When you change the prompt, bump PROMPT_VERSION. The harness will refuse to
append predictions if the hash doesn't match the file's recorded hash.
"""
import hashlib

# Human-readable version label - BUMP THIS when you change the prompt
PROMPT_VERSION = "v2-prose-only"

# The prompt template - matches production in main.py
# {text} will be replaced with the sample text (truncated to 2000 chars)
#
# v2 change: an explicit scope instruction, added after measuring that three
# lines of page furniture ("11 min read", a date, "Summarize with:") moved the
# score 50 points on identical prose -- from 85% down to 35%. With the furniture
# present, v1 classified the artifact ("web-scraped blog metadata, UI text");
# without it, v1 read the writing ("formulaic Key Takeaways, generic opening").
# Metadata is trivially forgeable, so anything the score depends on that is not
# the prose is a gaming vector. Everything else below is unchanged from v1 so the
# A/B isolates this instruction.
PROMPT_TEMPLATE = """Analyze the following text and determine if it appears to be AI-generated or human-written.

Judge the WRITING ONLY. This text was extracted from a web page and may still
contain page furniture: author names and bylines, publication or update dates,
reading times, breadcrumbs, navigation, share or summarize widgets, cookie
notices, and calls to action. Ignore all of it. Any page can carry those
elements, including a generated one, so they are not evidence of authorship.
Do not treat the presence of a byline, a date, or web UI text as a sign the
prose is human-written. Base your score only on the sentences themselves:
word choice, sentence rhythm, specificity, concreteness, and structure.

Return ONLY valid JSON with these exact fields:
- ai_probability_score: integer 0-100 (0 = definitely human, 100 = definitely AI)
- reasoning_flag: brief explanation string
- key_signals: array of strings with specific signals noticed

Text to analyze:
{text}

Respond with JSON only, no markdown or extra text."""

# Compute hash from template - automatic, catches silent edits
PROMPT_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:8]

# Model configuration
MODEL_NAME = "gemini-3.6-flash"
TEXT_TRUNCATION = 2000  # Same as production

# Concurrency settings
# Verify limits at: https://aistudio.google.com/rate-limit
# Tier 1 (paid, confirmed Aug 2026): 1,000 RPM · 10,000 RPD · 2M input TPM.
# 10 concurrent x ~6s per call = ~100 RPM, a 10x margin under the cap.
# On the free tier this was 5 RPM and had to run sequentially with a 13s delay.
MAX_CONCURRENT_REQUESTS = 10
REQUEST_DELAY_SECONDS = 0.1

# Pricing (for cost tracking in eval output)
GEMINI_INPUT_COST_PER_1M = 1.50
GEMINI_OUTPUT_COST_PER_1M = 7.50

# Dataset path
DEFAULT_DATASET_PATH = "../datasets/ai_vs_human_text_2026.csv"

# Default sample size for initial runs (prove pipeline before scaling)
# Start with 10 to work within 5 RPM rate limit
DEFAULT_SAMPLE_SIZE = 10

# Variance testing
VARIANCE_TEXTS = 30   # Number of texts to test
VARIANCE_TRIALS = 5   # Runs per text
