#!/usr/bin/env python3
"""Improved translation smoke-test for Sindhi to English PDF translation.

This script demonstrates the end-to-end flow for translating the first few
pages of a Sindhi PDF: text extraction (with optional OCR fallback), chunking,
translation through the OpenAI Responses API, and a simple PDF/text export of
the translated output.

Requirements (install via pip):
    python-dotenv, openai>=1.13, pypdf, pdf2image, pytesseract, reportlab

External dependency:
    Tesseract OCR binary must be available on PATH for pytesseract to work.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


# Constants
DEFAULT_PDF_MARGINS = 72  # points
DEFAULT_LINE_HEIGHT = 14  # points
DEFAULT_OCR_LANGUAGE = "snd"
SUPPORTED_REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high"]
SYSTEM_PROMPT = "You are a meticulous literary translator."
MAX_REASONING_RETRY_TOKENS = 16384
REASONING_BACKOFF = {
    "high": "medium",
    "medium": "low",
    "low": "minimal",
    "minimal": "none",
}

# Logger setup
_LOGGER = logging.getLogger(__name__)


@dataclass
class TranslationSettings:
    """Configuration for the translation process."""
    pdf_path: Path
    output_path: Path
    start_page: int
    max_pages: int
    chunk_chars: int
    chunk_overlap: int
    model: str
    temperature: Optional[float]
    max_output_tokens: int
    reasoning_effort: str
    
    def validate(self) -> None:
        """Validate settings and raise ValueError if invalid."""
        if self.chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be less than chunk_chars")
        if self.start_page <= 0:
            raise ValueError("start_page must be 1 or greater")
        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {SUPPORTED_REASONING_EFFORTS}")


class PDFExtractor:
    """Handles PDF text extraction with OCR fallback."""
    
    def __init__(self, ocr_lang: str = DEFAULT_OCR_LANGUAGE):
        self.ocr_lang = ocr_lang
    
    def extract_pages(self, pdf_path: Path, start_page: int, max_pages: int) -> List[str]:
        """Extract text from PDF pages, using OCR if necessary."""
        reader = PdfReader(str(pdf_path))
        texts: List[str] = []
        total_pages = len(reader.pages)
        start_index = max(0, start_page - 1)

        if start_index >= total_pages:
            raise ValueError(
                f"start_page {start_page} is beyond the total page count ({total_pages})."
            )

        if max_pages > 0:
            end_index = min(total_pages, start_index + max_pages)
        else:
            end_index = total_pages

        for index in range(start_index, end_index):
            page = reader.pages[index]
            text = (page.extract_text() or "").strip()

            if not self._contains_target_script(text):
                # Fallback to OCR when extraction misses Sindhi characters
                text = self._extract_with_ocr(pdf_path, index)

            texts.append(text)

        return texts

    @staticmethod
    def _contains_target_script(text: str) -> bool:
        """Check if the text already includes Sindhi/Arabic characters."""
        if not text:
            return False

        for char in text:
            code = ord(char)
            if (
                0x0600 <= code <= 0x06FF  # Arabic block
                or 0x0750 <= code <= 0x077F  # Arabic Supplement
                or 0x08A0 <= code <= 0x08FF  # Arabic Extended-A
            ):
                return True

        return False
    
    def _extract_with_ocr(self, pdf_path: Path, page_index: int) -> str:
        """Extract text from a single page using OCR."""
        _LOGGER.info("Running OCR on page %d", page_index + 1)
        
        images = convert_from_path(
            str(pdf_path),
            fmt="png",
            first_page=page_index + 1,
            last_page=page_index + 1,
            dpi=300,
        )
        
        ocr_texts = []
        for image in images:
            ocr_text = pytesseract.image_to_string(image, lang=self.ocr_lang)
            ocr_texts.append(ocr_text)
        
        return "\n".join(ocr_texts).strip()


class TextChunker:
    """Handles text chunking for API calls."""
    
    @staticmethod
    def chunk_text(text: str, chunk_chars: int, overlap: int) -> Iterator[str]:
        """Yield overlapping character-based chunks."""
        if not text:
            return
            
        position = 0
        length = len(text)
        step = max(1, chunk_chars - max(0, overlap))
        
        while position < length:
            end = min(length, position + chunk_chars)
            yield text[position:end]
            position += step


class TranslationService:
    """Handles translation via OpenAI API."""
    
    def __init__(self, client: OpenAI, settings: TranslationSettings):
        self.client = client
        self.settings = settings
    
    def translate_chunks(self, chunks: List[str]) -> List[str]:
        """Translate multiple text chunks."""
        translated: List[str] = []
        
        for index, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                continue
                
            _LOGGER.info("Translating chunk %d of %d", index, len(chunks))
            translated_text = self._translate_single_chunk(chunk, index)
            translated.append(translated_text)
        
        return translated
    
    def _translate_single_chunk(self, chunk: str, chunk_index: int) -> str:
        """Translate a single chunk of text."""
        prompt = self._create_translation_prompt(chunk)
        request_params = self._build_request_params(prompt)
        params: Dict[str, Any] = dict(request_params)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.responses.create(**params)
            except APIStatusError as exc:
                error_context = self._format_api_status_error(exc)
                _LOGGER.error(
                    "Translation failed for chunk %d (attempt %d/%d): %s",
                    chunk_index,
                    attempt,
                    max_attempts,
                    error_context,
                )
                raise RuntimeError(f"Translation failed for chunk {chunk_index}") from exc
            except Exception as exc:
                _LOGGER.error(
                    "Translation failed for chunk %d (attempt %d/%d): %s",
                    chunk_index,
                    attempt,
                    max_attempts,
                    exc,
                )
                raise RuntimeError(f"Translation failed for chunk {chunk_index}") from exc

            translated_text = self._extract_output_text(response)
            if translated_text:
                return translated_text

            retry_tokens = self._next_retry_tokens(params, response)
            if retry_tokens and attempt < max_attempts:
                _LOGGER.warning(
                    "Chunk %d used all %d tokens on reasoning; retrying with max_output_tokens=%d",
                    chunk_index,
                    params.get("max_output_tokens"),
                    retry_tokens,
                )
                params["max_output_tokens"] = retry_tokens
                continue

            if attempt < max_attempts:
                downgraded = self._downgrade_reasoning(params)
                if downgraded is not None:
                    if downgraded == "none":
                        _LOGGER.warning(
                            "Chunk %d still empty; retrying without reasoning",
                            chunk_index,
                        )
                    else:
                        _LOGGER.warning(
                            "Chunk %d still empty; lowering reasoning effort to %s",
                            chunk_index,
                            downgraded,
                        )
                    continue

            self._log_empty_response(chunk_index, response)
            raise RuntimeError(f"Empty translation for chunk {chunk_index}")

    def _create_translation_prompt(self, text: str) -> str:
        """Create the translation prompt."""
        return (
            "Translate the following Sindhi text into fluent English. "
            "Preserve names, cultural nuances, and paragraph breaks. "
            "Whenever the Sindhi surname عرساڻي appears, transliterate it as 'Ursani'.\n\n"
            + text.strip()
        )
    
    def _build_request_params(self, prompt: str) -> Dict[str, Any]:
        """Build parameters for the API request."""
        params = {
            "model": self.settings.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "max_output_tokens": self.settings.max_output_tokens,
        }

        if self.settings.temperature is not None:
            params["temperature"] = self.settings.temperature

        if self.settings.reasoning_effort != "none":
            params["reasoning"] = {"effort": self.settings.reasoning_effort}

        return params

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Return concatenated assistant text from a Responses API response."""
        output_text = getattr(response, "output_text", "")
        if output_text:
            return output_text.strip()

        texts: List[str] = []
        for output in getattr(response, "output", []) or []:
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text_segment = getattr(content, "text", "")
                    if text_segment:
                        texts.append(str(text_segment))

        return "".join(texts).strip()

    @staticmethod
    def _log_empty_response(chunk_index: int, response: Any) -> None:
        dumped = getattr(response, "model_dump", None)
        raw_dump = dumped() if callable(dumped) else str(response)
        response_dump = str(raw_dump)[:2000]
        _LOGGER.error(
            "Empty translation for chunk %d. Raw response: %s",
            chunk_index,
            response_dump,
        )

    @staticmethod
    def _next_retry_tokens(params: Dict[str, Any], response: Any) -> Optional[int]:
        requested_tokens = params.get("max_output_tokens")
        if not isinstance(requested_tokens, int):
            return None

        if requested_tokens >= MAX_REASONING_RETRY_TOKENS:
            return None

        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        output_tokens = getattr(usage, "output_tokens", None)
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None

        if not isinstance(output_tokens, int) or not isinstance(reasoning_tokens, int):
            return None

        if reasoning_tokens < requested_tokens:
            return None

        new_limit = min(
            MAX_REASONING_RETRY_TOKENS,
            max(requested_tokens * 2, requested_tokens + 1024),
        )

        if new_limit <= requested_tokens:
            return None

        return new_limit

    @staticmethod
    def _downgrade_reasoning(params: Dict[str, Any]) -> Optional[str]:
        reasoning = params.get("reasoning")
        if not isinstance(reasoning, dict):
            return None

        effort = reasoning.get("effort")
        next_effort = REASONING_BACKOFF.get(effort)
        if next_effort is None:
            return None

        if next_effort == "none":
            params.pop("reasoning", None)
            return "none"

        reasoning["effort"] = next_effort
        return next_effort

    @staticmethod
    def _format_api_status_error(exc: APIStatusError) -> str:
        """Return a concise string describing API 4xx/5xx responses."""
        details: List[str] = []

        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            details.append(f"status={status_code}")

        request_id = getattr(exc, "request_id", None)
        if request_id:
            details.append(f"request_id={request_id}")

        response = getattr(exc, "response", None)
        body_summary = TranslationService._summarize_response_body(response)
        if body_summary:
            details.append(body_summary)

        if not details:
            details.append(str(exc))

        return ", ".join(details)

    @staticmethod
    def _summarize_response_body(response: Any) -> Optional[str]:
        """Extract meaningful error message details from an httpx response."""
        if response is None:
            return None

        body: Any
        try:
            body = response.json()
        except Exception:
            try:
                body = response.text
            except Exception:
                body = None

        if isinstance(body, dict):
            error = body.get("error") or body
            if isinstance(error, dict):
                parts = []
                message = error.get("message")
                if message:
                    parts.append(f"message={TranslationService._trim_text(message)}")
                code = error.get("code")
                if code:
                    parts.append(f"code={code}")
                error_type = error.get("type")
                if error_type:
                    parts.append(f"type={error_type}")
                param = error.get("param")
                if param:
                    parts.append(f"param={param}")
                if parts:
                    return ", ".join(parts)

        if body:
            text = TranslationService._trim_text(str(body))
            if text:
                return f"body={text}"

        return None

    @staticmethod
    def _trim_text(text: str, limit: int = 500) -> str:
        """Trim whitespace and collapse multiline error messages."""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."


class OutputWriter:
    """Handles writing output to PDF or text files."""
    
    @staticmethod
    def write(text: str, destination: Path) -> Path:
        """Write text to the specified destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if destination.suffix.lower() == ".pdf":
            return OutputWriter._write_pdf(text, destination)
        else:
            return OutputWriter._write_text(text, destination)
    
    @staticmethod
    def _write_pdf(text: str, destination: Path) -> Path:
        """Write text to PDF."""
        pdf = canvas.Canvas(str(destination), pagesize=LETTER)
        width, height = LETTER
        x_margin = DEFAULT_PDF_MARGINS
        y_margin = DEFAULT_PDF_MARGINS
        y_position = height - y_margin
        
        for line in text.splitlines():
            if y_position < y_margin:
                pdf.showPage()
                y_position = height - y_margin
            
            pdf.drawString(x_margin, y_position, line)
            y_position -= DEFAULT_LINE_HEIGHT
        
        pdf.save()
        return destination
    
    @staticmethod
    def _write_text(text: str, destination: Path) -> Path:
        """Write text to a UTF-8 text file."""
        destination.write_text(text, encoding="utf-8")
        return destination


def load_environment() -> None:
    """Load environment variables and validate required settings."""
    load_dotenv(override=False)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Please set it in .env or as an environment variable."
        )


def parse_arguments() -> TranslationSettings:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate Sindhi PDF to English using OpenAI API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        type=Path,
        required=True,
        help="Path to the source Sindhi PDF"
    )
    
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=Path("data/output/translation.pdf"),
        help="Output path for translated content (PDF or TXT)"
    )

    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="First page number to translate (1-indexed)"
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=2,
        help="Maximum number of pages to translate (0 for all)"
    )
    
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=2000,
        help="Maximum characters per translation chunk"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use for translation"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0-2)"
    )
    
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=12000,
        help="Maximum tokens to generate per chunk"
    )
    
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=SUPPORTED_REASONING_EFFORTS,
        default="high",
        help="Reasoning effort level for capable models"
    )
    
    args = parser.parse_args()
    
    settings = TranslationSettings(
        pdf_path=args.pdf_path,
        output_path=args.output_path,
        start_page=args.start_page,
        max_pages=args.max_pages,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort
    )
    
    settings.validate()
    return settings


def main() -> None:
    """Main entry point for the translation script."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Parse arguments first so --help and validation work without API key
        settings = parse_arguments()
        load_environment()
        
        # Validate input file exists
        if not settings.pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {settings.pdf_path}")
        
        # Extract text from PDF
        _LOGGER.info(
            "Extracting text from %s (start page %d, max pages %s)",
            settings.pdf_path.name,
            settings.start_page,
            "all" if settings.max_pages == 0 else settings.max_pages,
        )
        extractor = PDFExtractor()
        page_texts = extractor.extract_pages(
            settings.pdf_path,
            settings.start_page,
            settings.max_pages,
        )
        
        # Combine and validate extracted text
        combined_text = "\n\n".join(filter(None, page_texts)).strip()
        if not combined_text:
            raise RuntimeError("No extractable text found in the PDF.")
        
        _LOGGER.info("Extracted %d characters of text", len(combined_text))
        
        # Chunk the text
        chunks = list(TextChunker.chunk_text(
            combined_text, 
            settings.chunk_chars, 
            settings.chunk_overlap
        ))
        _LOGGER.info("Created %d chunks for translation", len(chunks))
        
        # Translate chunks
        client = OpenAI()
        translator = TranslationService(client, settings)
        translations = translator.translate_chunks(chunks)
        
        # Combine translations
        full_translation = "\n\n".join(translations).strip()
        if not full_translation:
            raise RuntimeError("Translation resulted in empty output.")
        
        # Write output
        output_path = OutputWriter.write(full_translation, settings.output_path)
        _LOGGER.info("Successfully wrote translation to %s", output_path)
        
    except Exception as exc:
        _LOGGER.error("Translation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
