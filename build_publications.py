"""Build the bibliographic table for the VINITI submission."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from project_paths import ARTICLE_JSON_NAME, DATA_OUTPUT_DIR
from publication_metadata import (
    BIBLIOGRAPHIC_FIELDS,
    FIELDS,
    PUBLICATIONS_CSV,
    apply_publication_metadata,
    article_metadata,
    read_publication_rows,
    resolve_publications,
    write_article_json,
)


def _read_documents(source: Path):
    paths = [source]
    if source.is_dir():
        article_file = source / ARTICLE_JSON_NAME
        paths = [article_file] if article_file.is_file() else sorted(source.glob(f"*/{ARTICLE_JSON_NAME}"))

    for path in paths:
        yield path, _read_document(path)


def _read_document(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _update_article_file(path, row):
    document = _read_document(path)
    if not apply_publication_metadata(document, row):
        return False

    write_article_json(path, document)
    return True


def build_publications(source: str | Path = DATA_OUTPUT_DIR, output: str | Path = PUBLICATIONS_CSV,
                       *, update_json=False):
    """Build the CSV from saved parser outputs and optionally fill their bibliography."""
    articles = {path: article_metadata(document) for path, document in _read_documents(Path(source))}
    if not articles:
        raise ValueError(f"No {ARTICLE_JSON_NAME} files found in {source}")
    output = Path(output)
    previous = read_publication_rows(output)

    updated = resolve_publications(articles.values(), previous)
    merged = {**previous, **updated}
    rows = [merged[publication_id] for publication_id in sorted(merged)]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=output.parent, mode="w", newline="", encoding="utf-8", delete=False)
    path = Path(temporary.name)
    try:
        with temporary as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        path.replace(output)
    finally:
        path.unlink(missing_ok=True)

    if update_json:
        changed = sum(_update_article_file(path, updated[article["publication_id"]])
                      for path, article in articles.items())
        print(f"Updated bibliography in {changed} JSON files")

    for row in updated.values():
        missing = [field for field in BIBLIOGRAPHIC_FIELDS if not row[field]]
        if missing:
            print(f'{row["publication_id"]}: manual review for {", ".join(missing)}')
    print(f"Read {len(articles)} saved JSON files. Wrote {len(rows)} publications to {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build publications.csv from saved parser outputs.")
    parser.add_argument(
        "source", type=Path, nargs="?", default=DATA_OUTPUT_DIR,
        help=f"Path to {ARTICLE_JSON_NAME} or a parser output directory (default: {DATA_OUTPUT_DIR})",
    )
    parser.add_argument("-o", "--output", type=Path, default=PUBLICATIONS_CSV)
    parser.add_argument("--update-json", action="store_true",
                        help="Also fill bibliographic fields in the saved article JSON files")
    args = parser.parse_args()
    build_publications(args.source, args.output, update_json=args.update_json)


if __name__ == "__main__":
    main()
