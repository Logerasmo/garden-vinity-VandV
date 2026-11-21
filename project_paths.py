"""Paths shared by the PDF parser and the publication CSV builder."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TEST_PDF_DIR = PROJECT_DIR / "test"
DATA_OUTPUT_DIR = PROJECT_DIR / "data_output"
ARTICLE_JSON_NAME = "article.json"
