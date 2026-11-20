from pathlib import Path

import pymupdf

from super_neuronetwork_module import is_garbage


def parse_article(pdf_path, output_dir=None, garbage_path=None):
    pdf_path = Path(pdf_path)
    article_dir = Path("data_output") / pdf_path.stem
    useful_dir = Path(output_dir) if output_dir is not None else article_dir / "extracted_images"
    garbage_dir = Path(garbage_path) if garbage_path is not None else article_dir / "garbage"
    if useful_dir.resolve() == garbage_dir.resolve():
        raise ValueError("The useful and garbage image directories must differ.")

    useful_dir.mkdir(parents=True, exist_ok=True)
    garbage_dir.mkdir(parents=True, exist_ok=True)
    full_text = []
    images_by_digest = {}

    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            full_text.append(f"--- Страница {page_num} ---\n{text}")
            inline_images = None

            for info in page.get_image_info(hashes=True, xrefs=True):
                digest = info["digest"]
                if digest in images_by_digest:
                    record = images_by_digest[digest]
                    if record["pages"][-1] != page_num:
                        record["pages"].append(page_num)
                    continue

                if info["xref"]:
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
                garbage = is_garbage(image_bytes)
                filename = f"page{page_num}_{digest.hex()}.{image['ext']}"
                useful_file = useful_dir / filename
                garbage_file = garbage_dir / filename
                target, previous = (garbage_file, useful_file) if garbage else (useful_file, garbage_file)
                if previous.is_file() and previous.read_bytes() == image_bytes:
                    previous.replace(target)
                else:
                    target.write_bytes(image_bytes)

                images_by_digest[digest] = {
                    "page": page_num,
                    "pages": [page_num],
                    "file": str(target),
                    "is_garbage": garbage,
                }

    return "\n".join(full_text), list(images_by_digest.values())


def parse_all(pdf_dir):
    for path in sorted(Path(pdf_dir).iterdir()):
        if path.is_file() and path.suffix.lower() == ".pdf":
            article_text, _ = parse_article(path)
            file = Path("data_output") / path.stem / "article.txt"
            file.write_text(article_text, encoding="utf-8")


if __name__ == "__main__":
    parse_article("dev_articles/D022.pdf")

