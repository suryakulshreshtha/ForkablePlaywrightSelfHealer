"""Render the demo output as a PNG for articles and social cards.

A screenshot of a terminal carries whatever else was on screen - usernames,
home paths, pip notices. Rendering the captured text instead gives a clean,
high-resolution image with nothing leaked and consistent typography.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)          # github dark
FG = (201, 209, 217)
DIM = (110, 118, 129)
GREEN = (63, 185, 80)
AMBER = (210, 153, 34)
CYAN = (88, 166, 255)
WHITE = (240, 246, 252)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]


def _font(paths, size):
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def colourise(line: str):
    """Split a line into (text, colour) spans the way the CLI colours it."""
    if line.startswith("==="):
        return [(line, WHITE)]
    if line.startswith("PASSED"):
        return [("PASSED", GREEN), (line[6:], FG)]
    if line.strip().startswith("ok"):
        head, _, rest = line.partition("ok")
        spans = [(head, FG), ("ok", GREEN)]
        # Everything up to the selector column reads as normal text; the bound
        # selector is dim, and a [healed] tag is the one thing that should pull
        # the eye - it is the whole point of the image.
        marker = re.search(r"\s{4,}(testid|css|role|http)", rest)
        body, selector = (rest, "") if not marker else (rest[:marker.start()], rest[marker.start():])
        spans.append((body, FG))
        if "[healed]" in selector:
            before, tag, after = selector.partition("[healed]")
            spans += [(before, DIM), (tag, AMBER), (after, DIM)]
        elif selector:
            spans.append((selector, DIM))
        return spans
    return [(line, FG)]


def render(text: str, out: Path, size: int = 22, pad: int = 34) -> Path:
    font = _font(FONT_CANDIDATES, size)
    bold = _font(BOLD_CANDIDATES, size)
    lines = [ln.rstrip() for ln in text.strip("\n").splitlines()]

    char_w = font.getbbox("M")[2] or size * 0.6
    line_h = int(size * 1.62)
    width = int(pad * 2 + char_w * max((len(ln) for ln in lines), default=60) + 8)
    height = pad * 2 + line_h * len(lines) + 46

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # window chrome - three dots, so it reads as a terminal at thumbnail size
    for i, colour in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([pad + i * 26, 20, pad + i * 26 + 14, 34], fill=colour)
    draw.text((pad + 104, 18), "forkable demo", font=font, fill=DIM)
    draw.line([(0, 52), (width, 52)], fill=(33, 38, 45), width=1)

    y = pad + 34
    for line in lines:
        x = pad
        for span, colour in colourise(line):
            use = bold if colour is WHITE else font
            draw.text((x, y), span, font=use, fill=colour)
            x += draw.textlength(span, font=use)
        y += line_h

    img.save(out, optimize=True)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    captured = Path("/tmp/demo_raw.txt")
    if captured.exists():
        text = captured.read_text()
    else:
        text = subprocess.run(
            [sys.executable, "-m", "forkable_ai_agent", "demo", "--no-report"],
            capture_output=True, text=True, cwd=root,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "NO_COLOR": "1"},
        ).stdout
    # strip anything carrying a local filesystem path
    text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("report:"))

    out_dir = root / "docs"
    out_dir.mkdir(exist_ok=True)
    path = render(text, out_dir / "demo.png")
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
