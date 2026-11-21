import argparse
import json
import re
import sys
from collections import defaultdict
from hashlib import sha256
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from pdf_tables import extract_tables
from project_paths import ARTICLE_JSON_NAME, DATA_OUTPUT_DIR, PROJECT_DIR, TEST_PDF_DIR
from publication_metadata import (
    PUBLICATIONS_CSV,
    apply_publication_metadata,
    article_metadata,
    bibliography_from_metadata,
    missing_bibliography,
    read_publication_rows,
    resolve_publications,
    write_article_json,
)
from super_neuronetwork_module import is_garbage

EXTRACTED_IMAGES_DIR_NAME = "extracted_images"
UNCLASSIFIED_IMAGES_DIR_NAME = "images"
GARBAGE_DIR_NAME = "garbage"
PAGE_PREVIEWS_DIR_NAME = "pages"
ARTICLE_TEXT_NAME = "article.txt"
IMAGE_LAYOUT_VERSION = 5
GENERATED_IMAGE = re.compile(r"page\d+_(?:figure-[a-z0-9]+_\d+\.png|[a-f0-9]{32}(?:[a-f0-9]{32})?\.[a-z0-9]+)")
GENERATED_PREVIEW = re.compile(r"page\d+\.png")


class _ImageDraws(pymupdf.mupdf.FzDevice2):
    """Collect image placements and their active clipping rectangles from MuPDF."""

    def __init__(self, page_bounds):
        super().__init__()
        self.use_virtual_fill_image()
        self.use_virtual_fill_image_mask()
        self.page_bounds = page_bounds
        self.placements = defaultdict(list)

    def fill_image(self, ctx, image, matrix, alpha, color_params):
        transform = pymupdf.Matrix(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)
        bounds = pymupdf.Rect(0, 0, 1, 1) * transform
        scissor = self.fz_device_current_scissor()
        clip = pymupdf.Rect(scissor.x0, scissor.y0, scissor.x1, scissor.y1)
        visible = bounds & clip & self.page_bounds if alpha > 0 else pymupdf.Rect()
        self.placements[image.w, image.h].append((bounds, visible))

    def fill_image_mask(self, ctx, image, matrix, colorspace, color, alpha, color_params):
        self.fill_image(ctx, image, matrix, alpha, color_params)


def _bounds_difference(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def _visible_image_parts(page):
    images = page.get_image_info(hashes=True, xrefs=True)
    if not images:
        return []

    mupdf = pymupdf.mupdf
    device = _ImageDraws(page.rect)
    display = page.get_displaylist(annots=False)
    mupdf.fz_run_display_list(
        display.this, device, mupdf.FzMatrix(), mupdf.FzRect(*page.rect), mupdf.FzCookie()
    )
    parts = []
    for info in images:
        original = pymupdf.Rect(info["bbox"]) * page.rotation_matrix
        visible = original & page.rect
        placements = device.placements[info["width"], info["height"]]
        if placements:
            index = min(range(len(placements)), key=lambda i: _bounds_difference(original, placements[i][0]))
            if _bounds_difference(original, placements[index][0]) <= 1:
                _, visible = placements.pop(index)
        if visible.is_empty:
            continue

        bounds = visible * page.derotation_matrix
        if any(
            info["digest"] == other["digest"]
            and (bounds & other_bounds).get_area() >= 0.95 * (bounds | other_bounds).get_area()
            for other, other_bounds in parts
        ):
            continue
        parts.append((info, bounds))
    return parts


def _group_image_strips(page):
    images = _visible_image_parts(page)
    groups = []
    words = None

    for part in sorted(images, key=lambda item: (item[1].y0, item[1].x0)):
        _, lower = part
        matches = []
        for group in groups:
            _, upper = group[-1]
            gap = lower.y0 - upper.y1
            if not (-3 <= gap <= 3 and upper.y0 < lower.y0 and upper.y1 < lower.y1):
                continue

            left = max(upper.x0, lower.x0)
            right = min(upper.x1, lower.x1)
            if right - left < 0.9 * max(upper.width, lower.width):
                continue

            if gap > 0:
                if words is None:
                    words = page.get_text("words")
                if any(
                    x0 < right and x1 > left and upper.y1 <= (y0 + y1) / 2 <= lower.y0
                    for x0, y0, x1, y1, *_ in words
                ):
                    continue

            matches.append(group)

        if len(matches) == 1:
            matches[0].append(part)
        else:
            groups.append([part])

    return groups


def _render_image_group(page, group):
    bounds = pymupdf.Rect(group[0][1])
    for _, rect in group[1:]:
        bounds |= rect

    clip = bounds * page.rotation_matrix
    return page.get_pixmap(clip=clip, dpi=144, annots=False).tobytes("png")


def _save_figures(page, captions, figure_dir):
    figures = []
    for caption in captions:
        bounds = caption.get("figure_bbox")
        if bounds is None:
            continue
        figure_dir.mkdir(parents=True, exist_ok=True)
        label = caption["label"].lower().replace(" ", "-")
        filename = figure_dir / f"page{page.number + 1}_{label}_{len(figures) + 1}.png"
        clip = pymupdf.Rect(bounds) * page.rotation_matrix
        page.get_pixmap(clip=clip, dpi=144, annots=False).save(filename)
        figures.append({
            "label": caption["label"],
            "caption": caption["text"],
            "bbox": bounds,
            "file": str(filename),
        })
    return figures


def _enrich_bibliography(document, *, previous_path=None):
    if not missing_bibliography(document):
        return False

    publication_id = document["publication_id"]
    prepared = article_metadata(document)
    try:
        cached = read_publication_rows(DATA_OUTPUT_DIR / PUBLICATIONS_CSV.name).get(publication_id)
    except (OSError, KeyError, ValueError) as error:
        print(f"{document['source_file']}: could not read the publication cache ({error})", file=sys.stderr)
        cached = None

    if previous_path is not None:
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous["source_file"] == document["source_file"] and not missing_bibliography(previous):
                if apply_publication_metadata(document, bibliography_from_metadata(previous)):
                    return True
        except (OSError, ValueError, KeyError, TypeError):
            pass

    rows = {publication_id: cached} if cached else {}
    try:
        row = resolve_publications([prepared], rows)[publication_id]
    except (OSError, HTTPException, json.JSONDecodeError) as error:
        print(f"{document['source_file']}: Crossref lookup failed ({error})", file=sys.stderr)
        row = resolve_publications([prepared], rows, lookup=False)[publication_id]

    changed = apply_publication_metadata(document, row)
    missing = missing_bibliography(document)
    if missing:
        print(f"{document['source_file']}: bibliography needs review: {', '.join(missing)}", file=sys.stderr)
    return changed


def parse_pdf(pdf_path, output_dir=None, garbage_path=None, *, filter_images=True, article_dir=None):
    """Return all pages, tables, captions and image references for one PDF."""
    pdf_path = Path(pdf_path)
    article_dir = Path(article_dir) if article_dir is not None else DATA_OUTPUT_DIR / pdf_path.stem
    image_folder = EXTRACTED_IMAGES_DIR_NAME if filter_images else UNCLASSIFIED_IMAGES_DIR_NAME
    useful_dir = Path(output_dir) if output_dir is not None else article_dir / image_folder
    figure_dir = useful_dir if output_dir is not None else article_dir / EXTRACTED_IMAGES_DIR_NAME
    garbage_dir = Path(garbage_path) if garbage_path is not None else article_dir / GARBAGE_DIR_NAME
    if useful_dir.resolve() == garbage_dir.resolve():
        raise ValueError("The useful and garbage image directories must differ.")

    useful_dir.mkdir(parents=True, exist_ok=True)
    garbage_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    images_by_digest = {}

    with pymupdf.open(pdf_path) as doc:
        metadata = {key: value for key, value in (doc.metadata or {}).items() if value not in ("", None)}
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            groups = _group_image_strips(page)
            image_bounds = [box for group in groups for _, box in group]
            tables, captions = extract_tables(page, image_bounds=image_bounds)
            figures = _save_figures(page, captions, figure_dir)
            figure_regions = [pymupdf.Rect(figure["bbox"]) + (-0.25, -0.25, 0.25, 0.25)
                              for figure in figures]
            page_data = {
                "page": page_num,
                "text": text,
                "tables": tables,
                "captions": captions,
                "figures": figures,
                "image_ids": [],
                "needs_ocr": not text.strip(),
            }
            pages.append(page_data)
            inline_images = None

            if page_data["needs_ocr"]:
                preview_dir = useful_dir.parent / PAGE_PREVIEWS_DIR_NAME
                preview_dir.mkdir(parents=True, exist_ok=True)
                preview = preview_dir / f"page{page_num}.png"
                page.get_pixmap(dpi=144, annots=False).save(preview)
                page_data["preview_file"] = str(preview)

            for group in groups:
                # A saved figure already contains these raster panels and any PDF overlays.
                if any(all(figure.contains(box) for _, box in group) for figure in figure_regions):
                    continue
                info, bounds = group[0]
                digest = (
                    sha256(b"".join(part["digest"] for part, _ in group)).digest()
                    if len(group) > 1 else info["digest"]
                )
                image_id = digest.hex()
                region = pymupdf.Rect(bounds)
                for _, part_bounds in group[1:]:
                    region |= part_bounds
                placement = {"page": page_num, "bbox": [round(value, 2) for value in region]}

                if digest in images_by_digest:
                    record = images_by_digest[digest]
                    record["placements"].append(placement)
                    if record["pages"][-1] != page_num:
                        record["pages"].append(page_num)
                        page_data["image_ids"].append(image_id)
                    continue

                if len(group) > 1 or _bounds_difference(bounds, info["bbox"]) > 0.25:
                    image = {"image": _render_image_group(page, group), "ext": "png"}
                elif info["xref"]:
                    image = doc.extract_image(info["xref"])
                else:
                    if inline_images is None:
                        inline_images = {
                            block["number"]: block
                            for block in page.get_text("dict")["blocks"]
                            if block["type"] == 1
                        }
                    image = inline_images.get(info["number"])
                if not image:
                    continue

                image_bytes = image["image"]
                garbage = is_garbage(image_bytes) if filter_images else None
                filename = f"page{page_num}_{image_id}.{image['ext']}"
                useful_file = useful_dir / filename
                garbage_file = garbage_dir / filename
                target, previous = (garbage_file, useful_file) if garbage else (useful_file, garbage_file)
                if filter_images and previous.is_file() and previous.read_bytes() == image_bytes:
                    previous.replace(target)
                else:
                    target.write_bytes(image_bytes)

                images_by_digest[digest] = {
                    "id": image_id,
                    "page": page_num,
                    "pages": [page_num],
                    "placements": [placement],
                    "file": str(target),
                    "is_garbage": garbage,
                }
                page_data["image_ids"].append(image_id)

    document = {
        "publication_id": pdf_path.stem,
        "source_file": pdf_path.name,
        "image_layout_version": IMAGE_LAYOUT_VERSION,
        "metadata": metadata,
        "page_count": len(pages),
        "pages": pages,
        "images": list(images_by_digest.values()),
    }
    previous_path = DATA_OUTPUT_DIR / pdf_path.stem / ARTICLE_JSON_NAME
    _enrich_bibliography(document, previous_path=previous_path)
    return document


def _article_text(document):
    return "\n".join(f"--- Страница {page['page']} ---\n{page['text']}" for page in document["pages"])


def parse_article(pdf_path, output_dir=None, garbage_path=None):
    """Return the text and image records for callers using the original interface."""
    document = parse_pdf(pdf_path, output_dir, garbage_path)
    images = list(document["images"])
    for page in document["pages"]:
        number = page["page"]
        for figure in page["figures"]:
            images.append({
                "id": sha256(Path(figure["file"]).read_bytes()).hexdigest(),
                "page": number,
                "pages": [number],
                "placements": [{"page": number, "bbox": figure["bbox"]}],
                "file": figure["file"],
                "is_garbage": False,
            })
    return _article_text(document), images


def _publish_images(document, staging_dir, article_dir):
    paths = [image["file"] for image in document["images"]]
    for page in document["pages"]:
        paths.extend(figure["file"] for figure in page["figures"])
        if "preview_file" in page:
            paths.append(page["preview_file"])

    files = {Path(path) for path in paths}
    for file in files:
        target = article_dir / file
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging_dir / file).replace(target)
    return files


def _generated_image_files(article_dir):
    image_dirs = (EXTRACTED_IMAGES_DIR_NAME, UNCLASSIFIED_IMAGES_DIR_NAME, GARBAGE_DIR_NAME)
    for name in (*image_dirs, PAGE_PREVIEWS_DIR_NAME):
        folder = article_dir / name
        if not folder.is_dir():
            continue
        generated_name = GENERATED_PREVIEW if name == PAGE_PREVIEWS_DIR_NAME else GENERATED_IMAGE
        for path in folder.iterdir():
            if path.is_file() and generated_name.fullmatch(path.name):
                yield path.relative_to(article_dir)


def _save_article(pdf_path, *, filter_images):
    article_dir = DATA_OUTPUT_DIR / pdf_path.stem
    article_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=article_dir, prefix=".parse-") as scratch:
        staging_dir = Path(scratch)
        document = parse_pdf(pdf_path, article_dir=staging_dir, filter_images=filter_images)
        for image in document["images"]:
            image["file"] = Path(image["file"]).relative_to(staging_dir).as_posix()
        for page in document["pages"]:
            for figure in page["figures"]:
                figure["file"] = Path(figure["file"]).relative_to(staging_dir).as_posix()
            if "preview_file" in page:
                page["preview_file"] = Path(page["preview_file"]).relative_to(staging_dir).as_posix()

        (staging_dir / ARTICLE_TEXT_NAME).write_text(_article_text(document), encoding="utf-8")
        with (staging_dir / ARTICLE_JSON_NAME).open("w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        files = _publish_images(document, staging_dir, article_dir)
        (staging_dir / ARTICLE_TEXT_NAME).replace(article_dir / ARTICLE_TEXT_NAME)
        (staging_dir / ARTICLE_JSON_NAME).replace(article_dir / ARTICLE_JSON_NAME)

    for file in set(_generated_image_files(article_dir)) - files:
        (article_dir / file).unlink(missing_ok=True)
    print(f"{pdf_path.name} -> {article_dir / ARTICLE_JSON_NAME}")


def _completed_article(pdf_path, *, filter_images):
    article_dir = DATA_OUTPUT_DIR / pdf_path.stem
    try:
        document = json.loads((article_dir / ARTICLE_JSON_NAME).read_text(encoding="utf-8"))
        pages = document["pages"]
        if (
            document.get("image_layout_version") != IMAGE_LAYOUT_VERSION
            or document["source_file"] != pdf_path.name
            or document["page_count"] != len(pages)
            or not pages
            or not (article_dir / ARTICLE_TEXT_NAME).is_file()
        ):
            return None

        images = document["images"]
        if filter_images and any(image["is_garbage"] is None for image in images):
            return None

        files = [image["file"] for image in images]
        for page in pages:
            files.extend(figure["file"] for figure in page["figures"])
            if "preview_file" in page:
                files.append(page["preview_file"])
        if (all((article_dir / file).is_file() for file in files)
                and set(_generated_image_files(article_dir)) <= {Path(file) for file in files}):
            return document
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_all(pdf_dir=TEST_PDF_DIR, *, filter_images=True, resume=False):
    """Parse the PDF corpus, reporting per-file failures after the batch."""
    pdfs = sorted(path for path in Path(pdf_dir).iterdir()
                  if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    parsed = skipped = errors = 0
    for index, path in enumerate(pdfs, 1):
        print(f"[{index}/{len(pdfs)}] {path.name}", flush=True)
        try:
            document = _completed_article(path, filter_images=filter_images) if resume else None
            if document is not None:
                if _enrich_bibliography(document):
                    target = DATA_OUTPUT_DIR / path.stem / ARTICLE_JSON_NAME
                    write_article_json(target, document)
                    print(f"{path.name}: updated bibliography")
                skipped += 1
                continue
            _save_article(path, filter_images=filter_images)
        except Exception as error:
            print(f"{path.name}: {error}", file=sys.stderr, flush=True)
            errors += 1
        else:
            parsed += 1
    print(f"Total: {len(pdfs)} | Parsed: {parsed} | Existing: {skipped} | Errors: {errors}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Parse PDF pages and resolve publication metadata.")
    parser.add_argument("input", type=Path, nargs="?", default=TEST_PDF_DIR)
    parser.add_argument("--no-filter", action="store_true", help="Extract images without calling Ollama.")
    parser.add_argument("--resume", action="store_true", help="Reuse extracted pages and fill missing metadata in saved articles.")
    args = parser.parse_args()
    if args.input.is_dir():
        return int(bool(parse_all(args.input, filter_images=not args.no_filter, resume=args.resume)))
    if args.input.is_file() and args.input.suffix.lower() == ".pdf":
        _save_article(args.input, filter_images=not args.no_filter)
        return 0
    parser.error(f"PDF or folder not found: {args.input}")


if __name__ == "__main__":
    raise SystemExit(main())
