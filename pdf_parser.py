from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import pymupdf

from super_neuronetwork_module import is_garbage


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

            for group in _group_image_strips(page):
                info, bounds = group[0]
                digest = (
                    sha256(b"".join(part["digest"] for part, _ in group)).digest()
                    if len(group) > 1 else info["digest"]
                )
                if digest in images_by_digest:
                    record = images_by_digest[digest]
                    if record["pages"][-1] != page_num:
                        record["pages"].append(page_num)
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
        print(f"{path.name} done")


if __name__ == "__main__":
    parse_all("test")
