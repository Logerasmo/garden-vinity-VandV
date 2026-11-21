"""Extract scientific PDF tables from vector lines and positioned text."""

import re
from bisect import bisect_right

import pymupdf


_CAPTION = re.compile(r"^(Table|Figure|Fig\.?)\s*([A-Za-z]?\d+[A-Za-z]?)\s*[.:|]", re.I)
_NUMBER = re.compile(r"^[ (~≈<>+±−–-]*\d")
_SUPERSCRIPT = str.maketrans("0123456789-−+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁺")
_WILEY_MATH_GLYPHS = str.maketrans({1: "−", 3: "×"})


def _span_text(span):
    text = span["text"]
    if span["font"].endswith("AdvP4C4E74"):
        text = text.translate(_WILEY_MATH_GLYPHS)
    if span["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT:
        text = text.translate(_SUPERSCRIPT)
    return text


def _block_lines(block):
    for line in block.get("lines", ()):
        text = "".join(_span_text(span) for span in line["spans"]).strip()
        if text:
            yield pymupdf.Rect(line["bbox"]), text


def _lines(page, clip=None):
    for block in page.get_text("dict", clip=clip, flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        yield from _block_lines(block)


def _page_captions(blocks):
    for block in blocks:
        lines = list(_block_lines(block))
        if not lines:
            continue
        box, text = lines[0]
        match = _CAPTION.match(text)
        if not match:
            continue
        kind = "table" if match[1].lower() == "table" else "figure"
        caption_box = pymupdf.Rect(box)
        caption_text = [text]
        for line_box, line in lines[1:]:
            if _CAPTION.match(line) or line_box.y0 > caption_box.y1 + 8:
                break
            caption_box |= line_box
            caption_text.append(line)
        yield {
            "kind": kind,
            "label": f"{kind.title()} {match[2]}",
            "text": "\n".join(caption_text),
            "box": caption_box,
        }


def _drawing_paths(page):
    """Exclude paths outside their active Form or path clipping rectangles."""
    clips = []
    for path in page.get_drawings(extended=True):
        level = path["level"]
        while clips and clips[-1][0] >= level:
            clips.pop()
        if path["type"] == "clip":
            clips.append((level, path["scissor"]))
        elif path["type"] in {"s", "f", "fs"}:
            rect = path["rect"]
            if all(
                rect.x0 <= clip.x1 and rect.x1 >= clip.x0
                and rect.y0 <= clip.y1 and rect.y1 >= clip.y0
                for _, clip in clips
            ):
                yield path


def _horizontal_rules(paths):
    def rule(rect):
        if rect.width >= 65 and rect.height <= 2:
            return rect.x0, rect.x1, (rect.y0 + rect.y1) / 2
        return None

    for path in paths:
        whole_rule = rule(path["rect"])
        if whole_rule:
            yield whole_rule
            continue
        for item in path["items"]:
            if item[0] == "l":
                start, end = item[1:]
                segment = pymupdf.Rect(start, end).normalize()
            elif item[0] == "re":
                segment = item[1]
            else:
                continue
            line_rule = rule(segment)
            if line_rule:
                yield line_rule


def _column_breaks(lines, left, right):
    spans = []
    for box, _ in sorted(lines, key=lambda line: line[0].x0):
        if spans and box.x0 <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], box.x1)
        else:
            spans.append([box.x0, box.x1])
    if len(spans) < 2:
        return []
    return [left, *((first[1] + second[0]) / 2 for first, second in zip(spans, spans[1:])), right]


def _data_row_breaks(lines, columns):
    """Group cells on a common baseline and keep wrapped cell lines in that row."""
    anchors = []
    for box, text in lines:
        column = bisect_right(columns, (box.x0 + box.x1) / 2) - 1
        if 0 <= column < len(columns) - 1:
            anchors.append(((box.y0 + box.y1) / 2, box.y0, column, bool(_NUMBER.match(text))))

    groups = []
    for y, top, column, numeric in sorted(anchors):
        if not groups or y - groups[-1]["y"] > 3:
            groups.append({"y": y, "top": top, "columns": set(), "numeric_columns": set()})
        group = groups[-1]
        group["top"] = min(group["top"], top)
        group["columns"].add(column)
        if numeric and column > 0:
            group["numeric_columns"].add(column)

    minimum_columns = min(2, len(columns) - 2)
    starts = [group["top"] for group in groups
              if 0 in group["columns"] and len(group["columns"]) - 1 >= minimum_columns]
    if len(starts) < 2:
        starts = [group["top"] for group in groups if len(group["numeric_columns"]) >= minimum_columns]
    return [top - 1 for top in starts[1:]]


def _ruled_rows(page, x0, x1, horizontal):
    top, header_end, *_, bottom = horizontal
    bounds = pymupdf.Rect(x0 - 1, top - 1, x1 + 1, bottom + 1)
    lines = list(_lines(page, bounds))
    header = [(box, text) for box, text in lines if (box.y0 + box.y1) / 2 < header_end]
    columns = _column_breaks(header, bounds.x0, bounds.x1)
    if not columns:
        return None
    if len(horizontal) == 3:
        body = [(box, text) for box, text in lines if (box.y0 + box.y1) / 2 >= header_end]
        horizontal = [top, header_end, *_data_row_breaks(body, columns), bottom]

    cells = [[[] for _ in range(len(columns) - 1)] for _ in range(len(horizontal) - 1)]
    for box, text in sorted(lines, key=lambda line: (line[0].y0, line[0].x0)):
        row = bisect_right(horizontal, (box.y0 + box.y1) / 2) - 1
        column = bisect_right(columns, (box.x0 + box.x1) / 2) - 1
        if 0 <= row < len(cells) and 0 <= column < len(cells[row]):
            cells[row][column].append(text)

    rows = [["\n".join(cell) for cell in row] for row in cells if any(row)]
    if len(rows) < 2 or max(sum(bool(cell) for cell in row) for row in rows[1:]) < 2:
        return None
    return pymupdf.Rect(x0, top, x1, bottom), rows


def _find_ruled_table(page, caption, captions, rules):
    box = caption["box"]
    for x0, x1, top in rules:
        if top > box.y1 + 65:
            break
        if top < box.y1 - 1 or x0 >= box.x1 or x1 <= box.x0:
            continue

        limit = min(
            (other["box"].y0 - 6 for other in captions
             if other["box"].y0 > top and other["box"].x0 < x1 and other["box"].x1 > x0),
            default=page.rect.height,
        )
        horizontal = [top]
        for left, right, y in rules:
            if y > limit or (len(horizontal) >= 3 and y - horizontal[-1] > 165):
                break
            if y - horizontal[-1] >= 3 and abs(left - x0) <= 3 and abs(right - x1) <= 3:
                horizontal.append(y)
        if len(horizontal) >= 3:
            result = _ruled_rows(page, x0, x1, horizontal)
            if result:
                return result
    return None


def _record(bounds, rows, caption, method):
    return {
        "label": caption["label"] if caption else None,
        "caption": caption["text"] if caption else None,
        "bbox": [round(value, 2) for value in bounds],
        "rows": rows,
        "method": method,
    }


def _body_lines(blocks):
    """Keep short final lines with their paragraph when excluding prose from figures."""
    for lines in blocks:
        if not lines or _CAPTION.match(lines[0][1]):
            continue
        prose = [
            box for box, text in lines
            if box.width >= 120 and len(re.findall(r"[^\W\d_]{2,}", text)) >= 7
        ]
        if len(prose) >= 2:
            yield from (box for box, _ in lines)
        elif prose:
            yield prose[0]


def _overlaps_x(box, left, right):
    return box.x0 < right and box.x1 > left


def _caption_continues_in_right_column(label, center, text_blocks):
    if label.x1 > center + 8 or label.height < 16:
        return False
    for lines in text_blocks:
        if len(lines) < 2 or _CAPTION.match(lines[0][1]):
            continue
        first = lines[0][0]
        last = lines[-1][0]
        if (first.x0 >= center - 2 and abs(first.y0 - label.y0) <= 3
                and abs(last.y1 - label.y1) <= 12):
            return True
    return False


def _figure_regions(page, captions, drawings, text_blocks, body_lines, tables, image_bounds):
    """Locate artwork and its labels above captions, bounded by nearby prose and tables."""
    figure_captions = [caption for caption in captions if caption["kind"] == "figure"]
    if not figure_captions:
        return []

    page_box = page.rect * page.derotation_matrix
    table_boxes = [pymupdf.Rect(table["bbox"]) for table in tables if table["label"]]
    obstacles = [*table_boxes, *body_lines]
    padded_tables = [box + (-1, -1, 1, 1) for box in table_boxes]
    # Rect unions ignore zero-area paths, including many graph axes.
    artwork = []
    for path in drawings:
        box = path["rect"]
        if box.is_empty:
            box = box + (-0.5, -0.5, 0.5, 0.5)
        artwork.append(box)
    artwork.extend(image_bounds)
    artwork = [box for box in artwork if not any(table.contains(box) for table in padded_tables)]

    text_lines = [line for block in text_blocks for line in block]
    regions = []
    for caption in figure_captions:
        label_box = caption["box"]
        center = (page_box.x0 + page_box.x1) / 2
        left, right = page_box.x0 + 18, page_box.x1 - 18
        spans_columns = any(
            center - box.x0 >= page_box.width * 0.15 and box.x1 - center >= page_box.width * 0.15
            and 0 <= label_box.y0 - box.y1 <= 30
            and _overlaps_x(box, label_box.x0, label_box.x1)
            for box in image_bounds
        )
        if not spans_columns and not _caption_continues_in_right_column(label_box, center, text_blocks):
            if label_box.x0 >= center - 9:
                left = center - 2
            elif label_box.x1 <= center + 8:
                right = center + 2

        previous_caption_end = max(
            (other["box"].y1 for other in captions if other["box"].y1 < label_box.y0
             and _overlaps_x(other["box"], left, right)),
            default=page_box.y0 + 45,
        )
        upper_limit = max(previous_caption_end, page_box.y0 + 55)
        candidates = sorted(
            (box for box in artwork
             if max(box.width, box.height) >= 5 and box.y0 >= upper_limit
             and box.y1 <= label_box.y0 + 1 and box.x0 >= left - 2 and box.x1 <= right + 2
             and (box.height > 2 or box.width < (right - left) * 0.85)),
            key=lambda box: box.y1, reverse=True,
        )
        if not candidates or label_box.y0 - candidates[0].y1 > 55:
            continue

        bounds = pymupdf.Rect(candidates[0])
        for box in candidates[1:]:
            if box.y1 < bounds.y0 - 30:
                break
            if any(
                box.y0 < obstacle.y1 and obstacle.y0 < bounds.y0
                and _overlaps_x(obstacle, min(box.x0, bounds.x0), max(box.x1, bounds.x1))
                for obstacle in obstacles
            ):
                continue
            bounds |= box
        if bounds.width < 20 or bounds.height < 20:
            continue

        # Axis and panel labels can be separate PDF text just outside the artwork.
        left = max(left, bounds.x0 - 42)
        right = min(right, bounds.x1 + 42)
        preceding_content_end = max(
            (box.y1 for box in obstacles if box.y1 <= bounds.y0 + 1 and _overlaps_x(box, left, right)),
            default=previous_caption_end,
        )
        top = max(upper_limit, preceding_content_end)
        bottom = min(
            (box.y0 for box in obstacles if box.y0 >= bounds.y1 - 1 and _overlaps_x(box, left, right)),
            default=label_box.y0,
        )
        bottom = min(bottom, label_box.y0)
        text_area = pymupdf.Rect(left, max(top, bounds.y0 - 28), right, min(bottom, bounds.y1 + 28))
        panel_top = max(previous_caption_end, preceding_content_end, bounds.y0 - 28)
        panel_area = pymupdf.Rect(left, panel_top, right, min(bottom, bounds.y1 + 44))
        content = pymupdf.Rect(bounds)
        for box, text in text_lines:
            if text_area.contains(box) or (
                panel_area.contains(box) and re.fullmatch(r"(?:\([a-z]\)|[a-z]\)?)", text, re.I)
            ):
                content |= box
        crop = pymupdf.Rect(
            max(left, content.x0 - 2), max(panel_top, content.y0 - 2),
            min(right, content.x1 + 2), min(bottom, content.y1 + 2),
        )
        if crop.is_empty:
            continue
        caption["figure_bbox"] = [round(value, 2) for value in crop]
        regions.append(crop)
    return regions


def _is_plot_grid(bounds, figures, body_lines):
    overlap = max(((bounds & figure).get_area() for figure in figures), default=0)
    if overlap == 0:
        return False
    if overlap >= 0.7 * bounds.get_area():
        return True
    # A chart grid can join page rules and appear to contain scraps of a paragraph.
    cut_prose_lines = sum(
        bounds.y0 <= (box.y0 + box.y1) / 2 <= bounds.y1
        and _overlaps_x(box, bounds.x0, bounds.x1)
        and (box.x0 < bounds.x0 - 12 or box.x1 > bounds.x1 + 12)
        for box in body_lines
    )
    return cut_prose_lines >= 2


def extract_tables(page, *, image_bounds=None):
    """Return structured tables and the visible table/figure captions of one page."""
    blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
    text_blocks = [list(_block_lines(block)) for block in blocks]
    text_lines = [line for block in text_blocks for line in block]
    captions = list(_page_captions(blocks))
    drawings = list(_drawing_paths(page))
    if image_bounds is None:
        image_bounds = [pymupdf.Rect(image["bbox"]) for image in page.get_image_info()]
    table_captions = [caption for caption in captions if caption["kind"] == "table"]
    claimed_captions = set()
    result = []
    for table in page.find_tables(strategy="lines_strict", paths=drawings, use_layout=False).tables:
        rows = [row for row in table.extract() if any(row)]
        if len(rows) < 2 or table.col_count < 2 or not any(sum(bool(cell) for cell in row) >= 2 for row in rows):
            continue
        bounds = pymupdf.Rect(table.bbox)
        candidates = [
            index for index, caption in enumerate(table_captions)
            if index not in claimed_captions and -2 <= bounds.y0 - caption["box"].y1 <= 65
            and bounds.x0 < caption["box"].x1 and bounds.x1 > caption["box"].x0
        ]
        index = min(candidates, key=lambda i: abs(bounds.y0 - table_captions[i]["box"].y1), default=None)
        caption = table_captions[index] if index is not None else None
        if index is not None:
            claimed_captions.add(index)
        result.append(_record(bounds, rows, caption, "vector_grid"))

    rules = sorted(_horizontal_rules(drawings), key=lambda item: item[2])
    for index, caption in enumerate(table_captions):
        if index in claimed_captions:
            continue
        table = _find_ruled_table(page, caption, captions, rules)
        if table:
            bounds, rows = table
            result.append(_record(bounds, rows, caption, "horizontal_rules_and_text"))

    body_lines = list(_body_lines(text_blocks))
    figures = _figure_regions(page, captions, drawings, text_blocks, body_lines, result, image_bounds)
    result = [table for table in result
              if table["label"] or not _is_plot_grid(pymupdf.Rect(table["bbox"]), figures, body_lines)]
    result.sort(key=lambda table: (table["bbox"][1], table["bbox"][0]))
    for table in result:
        bounds = pymupdf.Rect(table["bbox"])
        table["notes"] = [
            text for box, text in text_lines
            if 0 <= box.y0 - bounds.y1 <= 30 and box.x0 < bounds.x1 and box.x1 > bounds.x0
            and text.lower().startswith(("illumination conditions:", "measurement conditions:"))
        ]
    for caption in captions:
        caption["bbox"] = [round(value, 2) for value in caption.pop("box")]
    return result, captions
