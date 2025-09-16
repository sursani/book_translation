# Repository Guidelines

## Project Structure & Module Organization
- `src/translate_improved.py` hosts the CLI entry point, translation pipeline, and helper classes; keep new modules in this package.
- `data/output/` stores generated PDFs or `.txt` transcripts; never check in bulky intermediates from local experiments.
- `data/tmp/` is safe for scratch OCR artifacts; clean it after runs to avoid stale page images.
- `requirements.txt` mirrors import usage; update it whenever imports change.

## Build, Test, and Development Commands
- `python3 -m venv venv && source venv/bin/activate` sets up an isolated interpreter (macOS/Linux; adjust for Windows).
- `pip install -r requirements.txt` installs runtime dependencies.
- `python src/translate_improved.py --pdf "$PWD/data/raw/book.pdf" --output data/output/sample.pdf --max-pages 2` runs a focused translation pass; prefer absolute paths for PDFs containing non-Latin characters.
- `python src/translate_improved.py --help` surfaces the full flag list when documenting new options.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, descriptive snake_case identifiers, and dataclass usage for configuration objects.
- Keep logging via `_LOGGER` instead of `print`; add `DEBUG` detail behind `--verbose` style flags rather than unconditional chatter.
- Document public helpers with short docstrings mirroring the existing tone, and prefer explicit type hints on function signatures.

## Testing Guidelines
- There is no automated suite yet; validate changes with a short `--max-pages` smoke run and keep the generated artifact in `data/output/` for review.
- When touching chunking or OCR logic, capture before/after excerpts in the PR description and compare token counts or OCR text lines.
- If adding unit tests, use `pytest` under `tests/` and name files `test_<feature>.py`; ensure they run via `pytest -q`.

## Commit & Pull Request Guidelines
- Follow the existing Git history: concise, imperative commit titles (e.g. `feat: Add Sindhi-to-English PDF translation tool` or `Update README and add Sukkur book PDF`).
- Squash work-in-progress commits before opening a PR; include context on model settings, page ranges, and sample command invocations.
- Link related issues, paste relevant log snippets, and attach OCR/translation diffs or screenshots when behavior changes.

## Security & Configuration Tips
- Store `OPENAI_API_KEY` in `.env` and load it through `python-dotenv`; never commit secrets or personal PDFs.
- Verify Tesseract availability with `tesseract --version` before running long jobs, and document any platform-specific configuration in PR notes.
