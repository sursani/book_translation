# Sindhi → English Translation (Current Script)

This repository contains a one-off Python script that translates a Sindhi-language PDF into English using the OpenAI Responses API. It supports text extraction with OCR fallback and writes the translated result to PDF (or `.txt` if you choose a `.txt` output path).

## Requirements
- Python 3.10+
- Tesseract OCR installed and available on PATH (required when pages have no selectable text)
  - Ensure the Sindhi language data (`snd`) is installed for Tesseract
- OpenAI API key

All Python dependencies are listed in `requirements.txt`, derived directly from `src/translate_improved.py`:
- `python-dotenv`
- `openai>=1.13`
- `pypdf`
- `pdf2image`
- `pytesseract`
- `reportlab`

## Setup
1. Create and activate a virtual environment
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     py -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR and ensure it’s on your PATH
   - macOS (Homebrew): `brew install tesseract`
   - Linux: use your package manager (`apt`, `dnf`, etc.)
   - Windows: install the Tesseract binary and add it to PATH
   - Add the Sindhi language data (`snd`) as needed for your platform
4. Configure your OpenAI API key
   - Create a `.env` file in the project root with:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage
Run the improved translation script:
```bash
python src/translate_improved.py \
  --pdf "/absolute/path/to/source.pdf" \
  --output data/output/translation.pdf \
  --max-pages 2 \
  --chunk-chars 2000 \
  --chunk-overlap 200 \
  --model gpt-5-high-reasoning \
  --temperature 0.2 \
  --max-output-tokens 4096 \
  --reasoning-effort high
```

Notes:
- Use an absolute path for `--pdf`, especially if the filename contains special characters.
- The script attempts normal PDF text extraction first; if a page has no selectable text, it falls back to OCR (requires Tesseract and `snd` language data).
- Set `--reasoning-effort none` when using non-reasoning models.
- Output is written as PDF if `--output` ends with `.pdf`; otherwise a UTF‑8 `.txt` file is written.

### Translating additional pages
- The script always starts at page 1 and processes as many pages as you specify with `--max-pages`. Increase the value to cover a larger portion of the book (e.g. `--max-pages 10` for the first ten pages).
- To translate the complete PDF in one pass, set `--max-pages 0`. The extractor will iterate through every page while chunking the text to fit model limits.
- For very long books you can run the script multiple times, saving each run to a different output file (`--output data/output/part1.txt`, `part2.txt`, etc.) with progressively larger `--max-pages` values so you keep checkpoints.
- If you decide to re-run only the newly added pages, delete or archive the previous output first so you don’t confuse versions.
- When using reasoning-optional models such as `gpt-4.1-mini`, pass `--model gpt-4.1-mini --reasoning-effort none` to avoid unsupported parameter errors.

## What the Script Does
1. Extracts text from the first N pages (configurable with `--max-pages`).
2. Chunks the text with controlled overlap to fit model limits.
3. Translates each chunk via the OpenAI Responses API.
4. Writes the combined translation to the specified output path.

## Current Project Layout
```
book_translation/
├── README.md
├── requirements.txt
├── src/
│   └── translate_improved.py
├── venv/                 # Created locally; not committed
└── <your source PDF files>
```

## References
- [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/text-generation)
- [pypdf documentation](https://pypdf.readthedocs.io/)
- [ReportLab user guide](https://www.reportlab.com/docs/reportlab-userguide.pdf)
  
