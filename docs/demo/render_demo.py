"""Turn a captured session into an mp4 (and gif) of the Slack conversation.

Input is `capture_supervised_slack.py`'s JSON: one entry per message the bot
posted or edited, with the timestamp it happened at and the buttons it carried.
Every character rendered here came out of the bot; this module chooses a
typeface and a background and nothing else.

    python docs/demo/render_demo.py docs/demo/session.json --out docs/media/x.mp4

Frames are drawn with Pillow and encoded with ffmpeg, both of which must be
available — this is a documentation tool, not part of the package, and neither
is a dependency of `grapharc` itself.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 800
MARGIN = 48
FPS = 12
#: Lines of a code fence one frame shows before it says how many it hid.
MAX_FENCE_LINES = 20

# Slack-ish dark palette. Not Slack's brand colours — this is a mock, and it
# should look like one rather than pass for a screenshot of the product.
BG = (26, 29, 33)
PANEL = (34, 37, 41)
FENCE_BG = (22, 24, 27)
TEXT = (222, 225, 230)
MUTED = (141, 148, 158)
ACCENT = (91, 140, 255)
GREEN = (46, 160, 106)
RED = (203, 68, 74)
AMBER = (222, 168, 62)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


class Renderer:
    def __init__(self) -> None:
        self.mono = _font("DejaVuSansMono.ttf", 15)
        self.mono_small = _font("DejaVuSansMono.ttf", 13)
        self.sans = _font("DejaVuSans.ttf", 17)
        self.sans_bold = _font("DejaVuSans-Bold.ttf", 17)
        self.title = _font("DejaVuSans-Bold.ttf", 21)
        self.small = _font("DejaVuSans.ttf", 13)

    # -- text helpers ---------------------------------------------------------

    def wrap(self, text: str, font, width: int) -> list[str]:
        """Wrap on words, then hard-break anything still too wide."""
        out: list[str] = []
        for raw in text.split("\n"):
            if not raw:
                out.append("")
                continue
            line = ""
            for word in raw.split(" "):
                candidate = f"{line} {word}".strip()
                if font.getlength(candidate) <= width or not line:
                    line = candidate
                else:
                    out.append(line)
                    line = word
            out.append(line)
        broken: list[str] = []
        for line in out:
            while font.getlength(line) > width and len(line) > 1:
                cut = len(line)
                while cut > 1 and font.getlength(line[:cut]) > width:
                    cut -= 1
                broken.append(line[:cut])
                line = line[cut:]
            broken.append(line)
        return broken

    # -- one frame ------------------------------------------------------------

    def _chrome(self, draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([0, 0, WIDTH, 64], fill=PANEL)
        draw.text((MARGIN, 20), "# incidents", font=self.title, fill=TEXT)
        draw.text(
            (WIDTH - MARGIN - 330, 24),
            "GraphARC — supervised Claude Code",
            font=self.small,
            fill=MUTED,
        )

    def _author(self, draw: ImageDraw.ImageDraw, y: int, *, app: bool) -> int:
        initials, name, colour = (
            ("ga", "grapharc", ACCENT) if app else ("yo", "you", (140, 110, 190))
        )
        draw.ellipse([MARGIN, y, MARGIN + 34, y + 34], fill=colour)
        draw.text((MARGIN + 9, y + 8), initials, font=self.small, fill=(255, 255, 255))
        draw.text((MARGIN + 48, y + 2), name, font=self.sans_bold, fill=TEXT)
        if app:
            draw.text((MARGIN + 140, y + 5), "APP", font=self.small, fill=MUTED)
        return y + 38

    def request_frame(self, request: str, caption: str) -> Image.Image:
        """The message a person typed. The one frame the bot did not produce."""
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        self._chrome(draw)
        y = self._author(draw, 96, app=False)
        for line in self.wrap(f"/grapharc {request}", self.mono, WIDTH - 2 * MARGIN - 48):
            draw.text((MARGIN + 48, y), line, font=self.mono, fill=TEXT)
            y += 22
        draw.rectangle([0, HEIGHT - 56, WIDTH, HEIGHT], fill=PANEL)
        draw.text((MARGIN, HEIGHT - 38), caption, font=self.small, fill=MUTED)
        return image

    def audit_frame(self, audit: dict, caption: str) -> Image.Image:
        """The closing frame: what the trace file says, not what a message said."""
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        self._chrome(draw)
        y = 110
        draw.text((MARGIN, y), "the trace, read back", font=self.title, fill=TEXT)
        y += 44
        order = " → ".join(audit.get("phase_order", []))
        rows = [
            ("trace", audit.get("trace", "—")),
            ("run", audit.get("run_id", "—")),
            ("events", str(audit.get("events", 0))),
            ("phase order", order or "—"),
            ("nodes that ran", ", ".join(audit.get("executed", [])) or "(none)"),
            ("tokens", str(audit.get("tokens", 0))),
            (
                "cost",
                "—" if audit.get("cost_usd") in (None, 0) else f"${audit['cost_usd']:.4f}",
            ),
        ]
        for label, value in rows:
            draw.text((MARGIN, y), f"{label:>16}", font=self.mono, fill=MUTED)
            lines = self.wrap(value, self.mono, WIDTH - 2 * MARGIN - 190) or [""]
            for index, line in enumerate(lines):
                draw.text((MARGIN + 180, y + index * 20), line, font=self.mono, fill=TEXT)
            y += 22 + 20 * (len(lines) - 1)
        y += 26
        verdict = audit.get("approved_before_first_node")
        draw.text(
            (MARGIN, y),
            "✓ approved before the first node started"
            if verdict
            else "✗ a node started before the approval was answered",
            font=self.sans_bold,
            fill=GREEN if verdict else RED,
        )
        draw.rectangle([0, HEIGHT - 56, WIDTH, HEIGHT], fill=PANEL)
        draw.text((MARGIN, HEIGHT - 38), caption, font=self.small, fill=MUTED)
        return image

    def frame(self, message: dict, *, caption: str, cursor: tuple | None = None) -> Image.Image:
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        self._chrome(draw)
        y = self._author(draw, 96, app=True)

        body_width = WIDTH - 2 * MARGIN - 48
        left = MARGIN + 48
        for block, kind in _blocks(message["text"]):
            if kind == "fence":
                lines = [
                    wrapped
                    for line in block.split("\n")
                    for wrapped in self.wrap(line, self.mono, body_width - 24)
                ]
                # Measured before the box is drawn, not while filling it: the
                # final message carries the CLI's whole report, and a rectangle
                # sized for every line of that runs off the bottom of the frame.
                if len(lines) > MAX_FENCE_LINES:
                    hidden = len(lines) - MAX_FENCE_LINES
                    lines = lines[:MAX_FENCE_LINES] + [f"… {hidden} more lines"]
                height = len(lines) * 20 + 20
                draw.rounded_rectangle(
                    [left, y, left + body_width, y + height], radius=6, fill=FENCE_BG
                )
                inner = y + 10
                for line in lines:
                    draw.text((left + 12, inner), line, font=self.mono, fill=_line_colour(line))
                    inner += 20
                y += height + 10
            else:
                for line in self.wrap(block, self.sans, body_width):
                    draw.text((left, y), line, font=self.sans, fill=_prose_colour(line))
                    y += 24
                y += 6
            if y > HEIGHT - 200:
                break

        if message.get("buttons"):
            y += 8
            x = left
            for index, label in enumerate(message["buttons"]):
                fill = GREEN if index == 0 else RED
                width = 120
                draw.rounded_rectangle([x, y, x + width, y + 42], radius=6, fill=fill)
                offset = (width - self.sans_bold.getlength(label)) / 2
                draw.text((x + offset, y + 10), label, font=self.sans_bold, fill=(255, 255, 255))
                x += width + 14
            if cursor is not None:
                cx, cy = left + cursor[0], y + cursor[1]
                draw.polygon(
                    [(cx, cy), (cx, cy + 20), (cx + 6, cy + 15), (cx + 13, cy + 26),
                     (cx + 18, cy + 23), (cx + 11, cy + 12), (cx + 18, cy + 11)],
                    fill=(255, 255, 255),
                    outline=(20, 20, 20),
                )

        draw.rectangle([0, HEIGHT - 56, WIDTH, HEIGHT], fill=PANEL)
        draw.text((MARGIN, HEIGHT - 38), caption, font=self.small, fill=MUTED)
        stamp = f"t+{message['at']:.1f}s"
        draw.text(
            (WIDTH - MARGIN - self.small.getlength(stamp), HEIGHT - 38),
            stamp,
            font=self.small,
            fill=MUTED,
        )
        return image


def _blocks(text: str) -> list[tuple[str, str]]:
    """Split a Slack message into prose and code-fence blocks."""
    out: list[tuple[str, str]] = []
    for index, chunk in enumerate(text.split("```")):
        if not chunk.strip():
            continue
        out.append((chunk.strip("\n"), "fence" if index % 2 else "prose"))
    return out


def _line_colour(line: str) -> tuple[int, int, int]:
    stripped = line.strip()
    if stripped.startswith("✎") or stripped.startswith("✗"):
        return AMBER if stripped.startswith("✎") else RED
    if stripped.startswith("✓"):
        return GREEN
    if stripped.startswith("▸"):
        return AMBER
    if "→" in stripped:
        return (168, 190, 230)
    return TEXT


def _prose_colour(line: str) -> tuple[int, int, int]:
    if "waiting for approval" in line:
        return AMBER
    if "nothing above has run yet" in line:
        return AMBER
    if line.startswith("<http"):
        return ACCENT
    return TEXT


#: Glyphs the bot emits that DejaVu has no outline for — they render as a
#: .notdef box, which reads as a rendering fault rather than as a mark. The
#: substitutes carry the same meaning in a shape this font can actually draw.
_SUBSTITUTES = {"⏸": "‖", "⬜": "▫"}


def _clean(text: str) -> str:
    """Strip Slack link markup so a rendered frame reads as text."""
    for glyph, substitute in _SUBSTITUTES.items():
        text = text.replace(glyph, substitute)
    text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 →", text)
    text = re.sub(r"<(https?://[^>]{0,60})[^>]*>", r"\1…", text)
    # The CLI writes the final state dump with escaped newlines inside it.
    text = text.replace("\\n", "\n")
    return text.replace("*", "")


CAPTIONS = {
    "start": "one Slack message — the gate has already forced --approve",
    "parked": "the proposed graph, in the message · NOTHING has run yet",
    "clicked": "a human clicks Approve — the decision is bound to the plan's fingerprint",
    "done": "only now does it run — and the trace records asked, answered, executed",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--hold", type=float, default=3.5, help="seconds per frame")
    parser.add_argument("--park-hold", type=float, default=6.0)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the video")

    session = json.loads(args.session.read_text(encoding="utf-8"))
    frames = [dict(f, text=_clean(f["text"])) for f in session["frames"]]
    if not frames:
        raise SystemExit("the session recorded no messages")

    renderer = Renderer()
    plan = []  # (image, seconds)
    # `--go` is what the person typed; `--approve` is what the gate added. The
    # video opens on the request precisely so that difference is visible.
    plan.append(
        (
            renderer.request_frame(
                session["request"], "what a person types — note: no --approve"
            ),
            args.hold,
        )
    )
    for index, message in enumerate(frames):
        parked = bool(message.get("buttons"))
        last = index == len(frames) - 1
        caption = CAPTIONS["start"] if index == 0 else (
            CAPTIONS["parked"] if parked else (CAPTIONS["done"] if last else "running…")
        )
        hold = args.park_hold if parked else args.hold
        plan.append((renderer.frame(message, caption=caption), hold))
        if parked:
            # The click: same frame, cursor drawn onto the Approve button.
            plan.append(
                (
                    renderer.frame(message, caption=CAPTIONS["clicked"], cursor=(52, 14)),
                    2.0,
                )
            )

    if session.get("audit"):
        plan.append(
            (
                renderer.audit_frame(
                    session["audit"], "the audit trail and the dashboard are one file"
                ),
                args.hold + 1.5,
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        listing = []
        for index, (image, seconds) in enumerate(plan):
            path = Path(tmp) / f"{index:04d}.png"
            image.save(path)
            listing.append(f"file '{path}'\nduration {seconds}")
        # ffmpeg's concat demuxer ignores the final entry's duration unless the
        # last file is repeated, which is why it appears twice.
        listing.append(f"file '{Path(tmp) / f'{len(plan) - 1:04d}.png'}'")
        script = Path(tmp) / "frames.txt"
        script.write_text("\n".join(listing), encoding="utf-8")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(script),
             "-vf", f"fps={FPS},format=yuv420p", "-movflags", "+faststart", str(args.out)],
            check=True, capture_output=True,
        )
        if args.gif is not None:
            palette = Path(tmp) / "palette.png"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(args.out), "-vf",
                 "fps=6,scale=880:-1:flags=lanczos,palettegen",
                 str(palette)], check=True, capture_output=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(args.out), "-i", str(palette), "-lavfi",
                 "fps=6,scale=880:-1:flags=lanczos[x];[x][1:v]paletteuse", str(args.gif)],
                check=True, capture_output=True,
            )
    print(f"{len(plan)} frames -> {args.out}" + (f" and {args.gif}" if args.gif else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
