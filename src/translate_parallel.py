#!/usr/bin/env python3
"""Parallel translation utility for Sindhi PDF to English."""

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from openai import OpenAI
from pypdf import PdfReader

from translate_improved import (
    TranslationSettings,
    PDFExtractor,
    TextChunker,
    TranslationService,
    OutputWriter,
    load_environment,
)


_LOGGER = logging.getLogger(__name__)


@dataclass
class BatchRange:
    """Describes a contiguous page range to process."""

    batch_id: int
    start_page: int
    end_page: int

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


@dataclass
class ParallelTranslationSettings:
    """Settings controlling the parallel translation workflow."""

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
    pages_per_job: int
    workers: int
    tmp_dir: Path
    keep_tmp: bool
    min_request_interval: float

    def validate(self) -> None:
        if self.start_page <= 0:
            raise ValueError("start_page must be 1 or greater")
        if self.max_pages < 0:
            raise ValueError("max_pages cannot be negative")
        if self.chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be less than chunk_chars")
        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.pages_per_job <= 0:
            raise ValueError("pages_per_job must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.min_request_interval < 0:
            raise ValueError("min_request_interval cannot be negative")

    def to_translation_settings(self, start_page: int, page_count: int) -> TranslationSettings:
        return TranslationSettings(
            pdf_path=self.pdf_path,
            output_path=self.output_path,
            start_page=start_page,
            max_pages=page_count,
            chunk_chars=self.chunk_chars,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
        )


class RateLimiter:
    """Simple shared rate limiter based on a minimum interval."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_ready = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_ready:
                    self._next_ready = now + self._min_interval
                    return
                delay = self._next_ready - now
            time.sleep(delay)


def compute_batch_ranges(
    total_pages: int,
    start_page: int,
    max_pages: int,
    pages_per_job: int,
) -> List[BatchRange]:
    """Split the requested page span into contiguous batches."""
    if start_page > total_pages:
        raise ValueError(
            f"start_page {start_page} is beyond the total page count ({total_pages})."
        )

    if max_pages == 0:
        last_page = total_pages
    else:
        last_page = min(total_pages, start_page + max_pages - 1)

    ranges: List[BatchRange] = []
    batch_id = 1
    current = start_page
    while current <= last_page:
        end_page = min(last_page, current + pages_per_job - 1)
        ranges.append(BatchRange(batch_id=batch_id, start_page=current, end_page=end_page))
        current = end_page + 1
        batch_id += 1

    return ranges


def translate_batch(
    batch: BatchRange,
    settings: ParallelTranslationSettings,
    limiter: RateLimiter,
) -> Path:
    """Translate a single batch of pages and persist the intermediate output."""
    extractor = PDFExtractor()
    page_texts = extractor.extract_pages(settings.pdf_path, batch.start_page, batch.page_count)
    combined_text = "\n\n".join(filter(None, page_texts)).strip()

    if not combined_text:
        _LOGGER.warning(
            "Batch %d (%d-%d) produced no extractable text; writing blank output",
            batch.batch_id,
            batch.start_page,
            batch.end_page,
        )
        tmp_path = settings.tmp_dir / f"batch_{batch.batch_id:04d}.txt"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text("", encoding="utf-8")
        return tmp_path

    chunks = list(
        TextChunker.chunk_text(
            combined_text,
            settings.chunk_chars,
            settings.chunk_overlap,
        )
    )

    _LOGGER.info(
        "Batch %d (%d-%d): translating %d chunks",
        batch.batch_id,
        batch.start_page,
        batch.end_page,
        len(chunks),
    )

    translation_settings = settings.to_translation_settings(
        start_page=batch.start_page,
        page_count=batch.page_count,
    )

    client = OpenAI()
    translator = TranslationService(client, translation_settings)

    translations: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            continue
        limiter.wait()
        _LOGGER.info(
            "Batch %d: translating chunk %d of %d",
            batch.batch_id,
            index,
            len(chunks),
        )
        translated_text = translator._translate_single_chunk(chunk, index)  # pylint: disable=protected-access
        translations.append(translated_text)

    full_translation = "\n\n".join(translations).strip()
    if not full_translation:
        raise RuntimeError(
            f"Translation for pages {batch.start_page}-{batch.end_page} produced empty output."
        )

    tmp_path = settings.tmp_dir / f"batch_{batch.batch_id:04d}.txt"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(full_translation, encoding="utf-8")

    return tmp_path


def gather_batch_outputs(batch_paths: Sequence[Tuple[BatchRange, Path]]) -> str:
    """Combine ordered batch outputs into a single translation string."""
    sorted_batches = sorted(batch_paths, key=lambda item: item[0].start_page)
    texts: List[str] = []
    has_content = False
    for batch, path in sorted_batches:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            _LOGGER.warning(
                "Merged output skipping empty batch %d (%d-%d)",
                batch.batch_id,
                batch.start_page,
                batch.end_page,
            )
            texts.append("")
            continue
        has_content = True
        texts.append(text)

    merged = "\n\n".join(filter(None, texts)).strip()

    if not has_content:
        raise RuntimeError("Translation produced no content across all batches.")

    return merged


def cleanup_temporary_files(batch_paths: Sequence[Tuple[BatchRange, Path]]) -> None:
    for _, path in batch_paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def parse_arguments() -> ParallelTranslationSettings:
    parser = argparse.ArgumentParser(
        description="Translate a Sindhi PDF in parallel using the OpenAI API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pdf", dest="pdf_path", type=Path, required=True, help="Path to the source PDF")
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=Path("data/output/translation_parallel.pdf"),
        help="Destination for the merged translation (PDF or TXT)",
    )
    parser.add_argument("--start-page", type=int, default=1, help="First page number to translate (1-indexed)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Total number of pages to translate (0 processes every page through the end of the document)",
    )
    parser.add_argument("--chunk-chars", type=int, default=2000, help="Maximum characters per translation chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between successive chunks")
    parser.add_argument("--model", type=str, default="gpt-5", help="OpenAI model to use")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omit to use the model default)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=12000,
        help="Maximum tokens to generate per chunk",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Reasoning effort level for capable models",
    )
    parser.add_argument(
        "--pages-per-job",
        type=int,
        default=2,
        help="Number of pages assigned to each parallel job",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum number of parallel translation workers",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("data/tmp/parallel"),
        help="Directory for intermediate batch outputs",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Retain intermediate batch files after completion",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds to wait between OpenAI API requests across all workers",
    )

    args = parser.parse_args()

    settings = ParallelTranslationSettings(
        pdf_path=args.pdf_path,
        output_path=args.output_path,
        start_page=args.start_page,
        max_pages=args.max_pages,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        pages_per_job=args.pages_per_job,
        workers=args.workers,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp,
        min_request_interval=args.min_request_interval,
    )

    settings.validate()
    return settings


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings: Optional[ParallelTranslationSettings] = None
    batch_results: List[Tuple[BatchRange, Path]] = []

    try:
        settings = parse_arguments()
        load_environment()

        if not settings.pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {settings.pdf_path}")

        pdf_reader = PdfReader(str(settings.pdf_path))
        total_pages = len(pdf_reader.pages)

        batches = compute_batch_ranges(
            total_pages=total_pages,
            start_page=settings.start_page,
            max_pages=settings.max_pages,
            pages_per_job=settings.pages_per_job,
        )

        if not batches:
            raise RuntimeError("No pages selected for translation.")

        limiter = RateLimiter(settings.min_request_interval)

        _LOGGER.info(
            "Translating %d batches across %d workers",
            len(batches),
            settings.workers,
        )

        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            future_to_batch = {
                executor.submit(translate_batch, batch, settings, limiter): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    tmp_path = future.result()
                    batch_results.append((batch, tmp_path))
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.error(
                        "Batch %d (%d-%d) failed: %s",
                        batch.batch_id,
                        batch.start_page,
                        batch.end_page,
                        exc,
                    )
                    raise

        merged_translation = gather_batch_outputs(batch_results)
        output_path = OutputWriter.write(merged_translation, settings.output_path)
        _LOGGER.info("Successfully wrote translation to %s", output_path)

        if settings.keep_tmp:
            for batch, tmp_path in batch_results:
                _LOGGER.info(
                    "Keeping temporary output for batch %d (%d-%d) at %s",
                    batch.batch_id,
                    batch.start_page,
                    batch.end_page,
                    tmp_path,
                )
        else:
            cleanup_temporary_files(batch_results)

    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Parallel translation failed: %s", exc)
        if settings and not settings.keep_tmp:
            cleanup_temporary_files(batch_results)
        sys.exit(1)


if __name__ == "__main__":
    main()
