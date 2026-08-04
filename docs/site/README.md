# GraphARC public site

Hand-authored static site — no npm, no build step. Deployed to GitHub Pages by
`.github/workflows/pages.yml`, which publishes **only this directory** (Pages
source must stay "GitHub Actions"; switching it to "deploy from branch /docs"
breaks the site and publishes the whole docs tree through Jekyll).

Preview locally:

```bash
python -m http.server 8080 --directory docs/site
```

All URLs are **relative** on purpose: production serves under the
`/GraphARC/` project-pages prefix, where a leading-slash URL 404s.

`assets/tokens.css` is the design-token sheet shared with the live run view
(`grapharc/server/static/view.css`) — change both together. `assets/brand/`
and `assets/media/` are copies of `docs/brand/` and `docs/media/` (the Pages
artifact is this directory alone, so it must be self-contained).
