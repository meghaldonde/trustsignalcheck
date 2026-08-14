#!/usr/bin/env python3
"""
Generate the SignalCheck icon set.

The mark is a shield (trust) with a signal pulse cut through it (the two
signals SignalCheck fuses: domain reputation + content authenticity).

Design notes
------------
* Geometry is authored once in a 512x512 design space and rasterised with
  8x supersampling, so every size is properly anti-aliased.
* Two optical variants. Below 48px the six-vertex pulse collapses into an
  unreadable smear, so small sizes get a simplified four-vertex pulse with
  a heavier stroke. This is deliberate, not a shortcut.
* Backgrounds are transparent. Chrome composites the toolbar icon over both
  light (#F1F3F4) and dark (#292A2D) chrome, so the shield silhouette --
  not a coloured tile -- has to carry the recognition.
* Pillow is the only dependency, so this runs anywhere without cairo or a
  headless browser. `signalcheck-mark.svg` is the canonical vector source
  and matches this geometry exactly.

Usage:  python scripts/generate_icons.py
"""

import math
import os

from PIL import Image, ImageDraw

# --------------------------------------------------------------------------
# Brand palette
# --------------------------------------------------------------------------
INDIGO_LIGHT = (79, 70, 229)     # #4F46E5  gradient start (top-left)
INDIGO_DARK = (49, 46, 129)      # #312E81  gradient end (bottom-right)
CYAN = (34, 211, 238)            # #22D3EE  accent: the rising signal
WHITE = (255, 255, 255)

# Design space. All geometry below is expressed in this square.
D = 512

# Sizes to emit. `pad` is transparent margin as a fraction of the canvas.
#   16/32 -> toolbar. Nearly full-bleed; every pixel counts.
#   48    -> chrome://extensions management page.
#   128   -> Chrome Web Store listing. The store expects the artwork to sit
#            in roughly the middle 96/128 of the canvas.
#   512   -> marketing / README / case study hero.
TARGETS = [
    (16, 0.02, "small"),
    (32, 0.02, "small"),
    (48, 0.04, "detail"),
    (128, 0.11, "detail"),
    (512, 0.11, "detail"),
]

SUPERSAMPLE = 8

# --------------------------------------------------------------------------
# Pulse geometry (design-space coordinates)
# --------------------------------------------------------------------------
# Full six-vertex pulse: flat baseline, small rise, deep drop, sharp recovery.
PULSE_DETAIL = [(78, 250), (150, 250), (196, 168), (240, 336),
                (286, 200), (330, 250), (434, 250)]
PULSE_DETAIL_SPLIT = 4          # index where white hands off to cyan
PULSE_DETAIL_STROKE = 36

# Simplified pulse for 16/32px: tails pulled in, one drop, one rise.
# Same rule as the detail variant -- white owns the spike, cyan owns the tail.
PULSE_SMALL = [(128, 252), (196, 252), (250, 348), (322, 152),
               (356, 202), (384, 252)]
PULSE_SMALL_SPLIT = 4
PULSE_SMALL_STROKE = 64


# --------------------------------------------------------------------------
# Shield outline
# --------------------------------------------------------------------------
def _cubic(p0, p1, p2, p3, steps=96):
    """Sample a cubic bezier into a point list."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u*u*t * p1[0] + 3 * u*t*t * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u*u*t * p1[1] + 3 * u*t*t * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _corner(cx, cy, r, a0, a1, steps=24):
    """Sample a circular arc (degrees, clockwise in screen space)."""
    return [
        (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        for a in (a0 + (a1 - a0) * i / steps for i in range(steps + 1))
    ]


def shield_outline(r=40, shoulder=272):
    """Shield filling the full 512x512 design space, top corners rounded."""
    pts = []
    pts += _corner(r, r, r, 180, 270)                    # top-left corner
    pts.append((D - r, 0))
    pts += _corner(D - r, r, r, 270, 360)                # top-right corner
    pts.append((D, shoulder))
    pts += _cubic((D, shoulder), (D, 388), (420, 470), (256, D))    # right flank
    pts += _cubic((256, D), (92, 470), (0, 388), (0, shoulder))     # left flank
    pts.append((0, r))
    return pts


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _gradient(size, c0, c1, base=64):
    """Diagonal (top-left -> bottom-right) linear gradient.

    Built small and upscaled: a linear ramp interpolates exactly, so this is
    identical to computing every pixel but orders of magnitude faster at the
    supersampled resolutions used here.
    """
    grad = Image.new("RGB", (base, base))
    px = grad.load()
    denom = 2 * (base - 1)
    for y in range(base):
        for x in range(base):
            t = (x + y) / denom
            px[x, y] = (
                round(c0[0] + (c1[0] - c0[0]) * t),
                round(c0[1] + (c1[1] - c0[1]) * t),
                round(c0[2] + (c1[2] - c0[2]) * t),
            )
    return grad.resize((size, size), Image.BICUBIC)


def _stroke(draw, pts, colour, width):
    """Polyline with round caps and round joins."""
    r = width / 2
    for a, b in zip(pts, pts[1:]):
        draw.line([a, b], fill=colour, width=int(round(width)))
    for x, y in pts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)


def render(size, pad, variant):
    """Render one icon at `size` px."""
    ss = size * SUPERSAMPLE
    scale = ss / D
    inner = 1.0 - 2 * pad
    off = ss * pad

    def T(p):
        return (p[0] * scale * inner + off, p[1] * scale * inner + off)

    # Shield mask, then paint the gradient through it.
    mask = Image.new("L", (ss, ss), 0)
    ImageDraw.Draw(mask).polygon([T(p) for p in shield_outline()], fill=255)

    canvas = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    canvas.paste(_gradient(ss, INDIGO_LIGHT, INDIGO_DARK).convert("RGBA"), (0, 0), mask)

    # Pulse, drawn on its own layer then clipped to the shield so the round
    # caps can never spill past the silhouette.
    if variant == "small":
        pulse, split, stroke = PULSE_SMALL, PULSE_SMALL_SPLIT, PULSE_SMALL_STROKE
    else:
        pulse, split, stroke = PULSE_DETAIL, PULSE_DETAIL_SPLIT, PULSE_DETAIL_STROKE

    w = stroke * scale * inner
    layer = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    _stroke(ld, [T(p) for p in pulse[:split + 1]], WHITE + (255,), w)
    _stroke(ld, [T(p) for p in pulse[split:]], CYAN + (255,), w)
    canvas = Image.alpha_composite(canvas, layer)
    canvas.putalpha(Image.composite(canvas.getchannel("A"),
                                    Image.new("L", (ss, ss), 0), mask))

    return canvas.resize((size, size), Image.LANCZOS)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    icons_dir = os.path.join(root, "extension", "icons")
    os.makedirs(icons_dir, exist_ok=True)

    for size, pad, variant in TARGETS:
        img = render(size, pad, variant)
        out = os.path.join(icons_dir, f"icon{size}.png")
        img.save(out, "PNG", optimize=True)
        print(f"  {os.path.relpath(out, root)}  ({variant} variant)")

    print(f"\nWrote {len(TARGETS)} icons to {os.path.relpath(icons_dir, root)}")


if __name__ == "__main__":
    main()
