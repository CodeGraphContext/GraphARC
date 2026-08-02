# GraphARC brand assets

The mark is an open arc — the bounded ceiling a governed run executes under, and
literally the *ARC* — wrapping a three-node graph, the *Graph*. One shape for
both halves of the name; the gap at the top keeps it from reading as a generic
ring.

Wordmark is Poppins (Medium for `Graph`, Bold for `ARC`, so the gradient lands
on the emphasized half). **The type is converted to vector outlines, not `<text>`
elements**, so every SVG here renders identically without Poppins installed.

## Which file to use

| File | Use for |
| --- | --- |
| `grapharc-logo-ondark.svg` | Lockup on a dark background (transparent) — this is the README's dark-mode asset |
| `grapharc-logo-light.svg` | Lockup on a light background (transparent) |
| `grapharc-logo-dark.svg` | Lockup with its own dark rounded panel, for placing on an arbitrary background |
| `grapharc-icon-transparent.svg` | Icon alone, no background |
| `grapharc-icon-dark.svg` | Icon on a dark rounded square — social avatars, app icons |
| `grapharc-favicon.svg` + `grapharc-favicon-*.png` | Small sizes only (see below) |

`.png` siblings are 3x raster exports of the same artwork, for contexts that
cannot take SVG. Prefer the SVG wherever it is accepted.

## Use the favicon file below ~48px

`grapharc-favicon.svg` is not the same artwork scaled down — it carries heavier
arc and edge strokes and larger nodes, because the standard icon's inner
triangle turns to mush at favicon sizes. Rasters are provided at 16, 32, 64, 180
(Apple touch icon) and 512.

## Palette

The gradient runs teal → blue → violet:

| Stop | Icon arc | Inner graph (lighter, for contrast against the arc) |
| --- | --- | --- |
| 0% | `#2DD4BF` | `#5EEAD4` |
| 50–55% | `#3B82F6` | `#60A5FA` |
| 100% | `#8B5CF6` | `#A78BFA` |

Dark background: `#0B0F19`. Wordmark: `#E7ECF3` on dark, `#0F172A` on light.

## If you edit these

Gradients use `gradientUnits="userSpaceOnUse"`, and that is deliberate rather
than incidental. Under the default `objectBoundingBox`, a perfectly horizontal
or vertical stroke has a zero-height (or zero-width) bounding box, which makes
the gradient undefined — so the element silently does not paint at all. The
triangle's top edge is exactly that case, and it vanished from the first render
of this mark for that reason. Keep the user-space units, or that edge disappears
again with no error anywhere.
