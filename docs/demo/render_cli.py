"""Render a captured terminal session to an mp4 (and gif).

The recording holds real bytes with real ANSI colour; this draws them into a
terminal-shaped frame sequence. The CLI emits SGR colour and nothing else — no
cursor addressing, no spinners — so a full terminal emulator is not needed and
would be a worse thing to trust: what is parsed here is exactly `\\x1b[…m`, and
any other escape is dropped rather than half-interpreted.

Pacing is the one thing this invents. A command is typed out, its output is
revealed a few lines at a time, and each step holds long enough to read. Real
durations are not thrown away — each step's frame shows the wall clock the
command actually took, so a 43-second agent phase says so while still being
watchable.

    python docs/demo/render_cli.py session.json --out docs/media/x.mp4 --gif x.gif
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

COLS, ROWS = 100, 30
FONT_SIZE = 17
#: Measured from the font rather than chosen, so a column lands exactly where
#: the glyph before it ends. A guessed cell width shows up as text that either
#: overlaps or drifts apart across a 100-column line.
CELL_W = 10.234375  # DejaVuSansMono advance at FONT_SIZE
CELL_H = 22
PAD = 26
HEADER = 44
FOOTER = 46


def _even(value: float) -> int:
    """h264's yuv420p subsamples by two, so an odd dimension is an encode error."""
    return round(value / 2) * 2


WIDTH = _even(COLS * CELL_W + 2 * PAD)
HEIGHT = _even(ROWS * CELL_H + 2 * PAD + HEADER + FOOTER)
FPS = 12
#: A command is typed in this many frames whatever its length, over this many
#: seconds. Fixed frames rather than fixed characters-per-frame: a
#: `grapharc plan '<goal>' --model … --registry …` line runs past 250
#: characters, and three-characters-a-frame spent eighty frames — seven
#: seconds — watching a prompt fill in.
TYPING_FRAMES = 10
TYPING_SECONDS = 1.1

BG = (18, 20, 24)
CHROME = (30, 33, 38)
FG = (214, 219, 226)
MUTED = (132, 140, 152)
PROMPT = (110, 190, 130)

#: The sixteen named ANSI colours. `cli/style.py` picks its depth from the
#: terminal and emits `38;5;N` under `TERM=xterm-256color` (which is what the
#: capture sets), so these are the fallback for a 16-colour recording.
ANSI_16 = [
    (40, 44, 52), (222, 95, 95), (110, 190, 130), (222, 176, 92),
    (108, 158, 232), (186, 130, 220), (94, 190, 200), FG,
    MUTED, (240, 130, 130), (140, 210, 155), (238, 200, 120),
    (140, 180, 240), (208, 160, 235), (130, 210, 218), (245, 248, 252),
]


def xterm256(index: int) -> tuple[int, int, int]:
    """Colour `N` of `38;5;N`, by the standard's own construction.

    Computed rather than tabulated: 0-15 are the named colours, 16-231 are a
    6x6x6 cube with the documented non-linear ramp, 232-255 are the grey
    ramp. A hand-copied 256-entry table is exactly the kind of thing that ends
    up subtly wrong and makes a demo misrepresent what the tool printed.
    """
    if index < 16:
        return ANSI_16[index]
    if index < 232:
        index -= 16
        levels = (0, 95, 135, 175, 215, 255)
        return (levels[index // 36], levels[(index // 6) % 6], levels[index % 6])
    value = 8 + (index - 232) * 10
    return (value, value, value)


FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

#: One pattern for every escape, classified by its final byte — not two
#: patterns racing each other. A separate "strip the ones I don't handle" regex
#: matched `\x1b[38;5;245m` too (its final byte is a letter like any other) and
#: silently removed every colour before the SGR parser ran, which renders a
#: perfectly colourful session in monochrome.
_CSI = re.compile(r"\x1b\[([0-9;?]*)([@-~])|\x1b\][^\x07]*\x07")


class Screen:
    """A fixed grid of (char, colour, bold) cells that scrolls."""

    def __init__(self) -> None:
        self.rows: list[list[tuple[str, tuple, bool]]] = [[]]
        self.colour = FG
        self.bold = False

    def write(self, text: str) -> None:
        position = 0
        for match in _CSI.finditer(text):
            self._plain(text[position : match.start()])
            # Only `m` is colour. Every other CSI (and any OSC) is dropped
            # rather than half-interpreted: the CLI emits none of them, so
            # guessing at one would be inventing behaviour to render.
            if match.group(2) == "m":
                self._sgr(match.group(1))
            position = match.end()
        self._plain(text[position:])

    def _sgr(self, params: str) -> None:
        codes = [int(raw or 0) for raw in (params or "0").split(";")]
        index = 0
        while index < len(codes):
            code = codes[index]
            if code == 0:
                self.colour, self.bold = FG, False
            elif code == 1:
                self.bold = True
            elif code == 2:
                self.colour = MUTED
            elif code == 38 and index + 1 < len(codes):
                # Extended foreground: `38;5;N` (256) or `38;2;R;G;B` (true
                # colour). Both consume their arguments, so the loop has to
                # skip them — reading them as further SGR codes would repaint
                # the text with whatever the channel values happened to mean.
                if codes[index + 1] == 5 and index + 2 < len(codes):
                    self.colour = xterm256(codes[index + 2])
                    index += 2
                elif codes[index + 1] == 2 and index + 4 < len(codes):
                    self.colour = tuple(codes[index + 2 : index + 5])
                    index += 4
            elif code == 39:
                self.colour = FG
            elif 30 <= code <= 37:
                self.colour = ANSI_16[code - 30]
            elif 90 <= code <= 97:
                self.colour = ANSI_16[code - 90 + 8]
            index += 1

    def _plain(self, text: str) -> None:
        for char in text:
            if char == "\n":
                self.rows.append([])
            elif char == "\r":
                self.rows[-1] = []
            elif char == "\t":
                self._plain(" " * (4 - len(self.rows[-1]) % 4))
            elif char >= " ":
                if len(self.rows[-1]) >= COLS:
                    self.rows.append([])
                self.rows[-1].append((char, self.colour, self.bold))

    def visible(self) -> list[list[tuple[str, tuple, bool]]]:
        return self.rows[-ROWS:]

    def clone(self) -> Screen:
        copy = Screen()
        copy.rows = [list(row) for row in self.rows]
        copy.colour, copy.bold = self.colour, self.bold
        return copy


class Renderer:
    def __init__(self, title: str, subtitle: str) -> None:
        self.mono = ImageFont.truetype(str(FONT_DIR / "DejaVuSansMono.ttf"), FONT_SIZE)
        self.mono_bold = ImageFont.truetype(
            str(FONT_DIR / "DejaVuSansMono-Bold.ttf"), FONT_SIZE
        )
        self.sans = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 14)
        self.sans_bold = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 15)
        self.title = title
        self.subtitle = subtitle

    def draw(self, screen: Screen, *, caption: str, elapsed: str = "") -> Image.Image:
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)

        draw.rectangle([0, 0, WIDTH, HEADER], fill=CHROME)
        for index, colour in enumerate(((236, 106, 94), (238, 190, 82), (108, 198, 118))):
            x = PAD + index * 20
            draw.ellipse([x, 16, x + 12, 28], fill=colour)
        draw.text((PAD + 80, 14), self.title, font=self.sans_bold, fill=FG)
        if self.subtitle:
            offset = PAD + 92 + self.sans_bold.getlength(self.title)
            draw.text((offset, 15), self.subtitle, font=self.sans, fill=MUTED)

        top = HEADER + PAD
        for row_index, row in enumerate(screen.visible()):
            y = top + row_index * CELL_H
            for column, (char, colour, bold) in enumerate(row):
                draw.text(
                    (PAD + column * CELL_W, y),
                    char,
                    font=self.mono_bold if bold else self.mono,
                    fill=colour,
                )

        draw.rectangle([0, HEIGHT - FOOTER, WIDTH, HEIGHT], fill=CHROME)
        draw.text((PAD, HEIGHT - FOOTER + 15), caption, font=self.sans, fill=MUTED)
        if elapsed:
            draw.text(
                (WIDTH - PAD - self.sans.getlength(elapsed), HEIGHT - FOOTER + 15),
                elapsed,
                font=self.sans,
                fill=MUTED,
            )
        return image


def _prompt_line(command: str, upto: int) -> str:
    return f"\x1b[32m$\x1b[0m {command[:upto]}"


def build_frames(session: dict, args) -> list[tuple[Image.Image, float]]:
    renderer = Renderer(session["title"], session.get("subtitle", ""))
    screen = Screen()
    frames: list[tuple[Image.Image, float]] = []

    for step in session["steps"]:
        command = step["command"]
        caption = step.get("caption") or ""
        elapsed = f"{step['seconds']:.1f}s  ·  exit {step['exit_code']}"

        # Type the command in a fixed number of frames rather than a fixed
        # number of characters per frame. A `grapharc plan '<goal>' --model …`
        # line runs past 250 characters, and one frame per three characters
        # made *typing it* eighty frames — seven seconds of watching a prompt
        # fill in, and the single biggest contributor to the file size.
        base = screen.clone()
        steps = max(1, min(TYPING_FRAMES, len(command)))
        for frame_index in range(1, steps + 1):
            typing = base.clone()
            typing.write(_prompt_line(command, round(len(command) * frame_index / steps)))
            frames.append((renderer.draw(typing, caption=caption), TYPING_SECONDS / steps))
        screen.write(_prompt_line(command, len(command)) + "\n")
        frames.append((renderer.draw(screen, caption=caption), 0.35))

        # Reveal the output in blocks, so a long report scrolls rather than
        # appearing whole.
        lines = step["output"].splitlines()
        block = max(1, args.lines_per_frame)
        for index in range(0, len(lines), block):
            screen.write("\n".join(lines[index : index + block]) + "\n")
            frames.append((renderer.draw(screen, caption=caption, elapsed=elapsed), 1 / FPS * 2))
        hold = step.get("hold") or args.hold
        frames.append((renderer.draw(screen, caption=caption, elapsed=elapsed), hold))
        screen.write("\n")

    return frames


def encode(frames, out: Path, gif: Path | None) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the video")
    with tempfile.TemporaryDirectory() as tmp:
        listing = []
        for index, (image, seconds) in enumerate(frames):
            path = Path(tmp) / f"{index:05d}.png"
            image.save(path)
            listing.append(f"file '{path}'\nduration {seconds}")
        listing.append(f"file '{Path(tmp) / f'{len(frames) - 1:05d}.png'}'")
        script = Path(tmp) / "frames.txt"
        script.write_text("\n".join(listing), encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(script),
             "-vf", f"fps={FPS},format=yuv420p", "-movflags", "+faststart", str(out)],
            check=True, capture_output=True,
        )
        if gif is not None:
            palette = Path(tmp) / "palette.png"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(out), "-vf",
                 "fps=6,scale=900:-1:flags=lanczos,palettegen", str(palette)],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(out), "-i", str(palette), "-lavfi",
                 "fps=6,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse", str(gif)],
                check=True, capture_output=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--hold", type=float, default=2.6, help="seconds at the end of a step")
    parser.add_argument("--lines-per-frame", type=int, default=2)
    args = parser.parse_args()

    session = json.loads(args.session.read_text(encoding="utf-8"))
    frames = build_frames(session, args)
    encode(frames, args.out, args.gif)
    total = sum(seconds for _image, seconds in frames)
    where = f"{args.out}" + (f" and {args.gif}" if args.gif else "")
    print(f"{len(frames)} frames, {total:.1f}s -> {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
