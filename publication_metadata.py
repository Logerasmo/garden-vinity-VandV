"""Resolve and apply publication metadata for the PDF parser and CSV export."""

import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from project_paths import DATA_OUTPUT_DIR

PUBLICATIONS_CSV = DATA_OUTPUT_DIR / "publications.csv"
FIELDS = ("publication_id", "title", "year", "DOI", "authors", "in_scope", "exclusion_reason")
BIBLIOGRAPHIC_FIELDS = FIELDS[1:5]
CROSSREF_URL = "https://api.crossref.org/works"
CROSSREF_FIELDS = "DOI,title,author,published,published-print,published-online,issued"
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE | re.ASCII)


def _clean(text):
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    return " ".join(unescape(text).split())


def _key(text):
    return "".join(character for character in unicodedata.normalize("NFKD", text.casefold())
                   if character.isalnum())


def _header_score(title_key, headers):
    """Match ordered title fragments around text inserted by PDF column ordering."""
    scores = []
    for header in headers:
        matcher = SequenceMatcher(None, title_key, header, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks() if block.size >= 8)
        scores.append(matched / len(title_key))
    return max(scores, default=0)


def _dois(text):
    for match in DOI_PATTERN.finditer(text):
        doi = match[0].rstrip(".,;:")
        while doi.endswith(")") and doi.count(")") > doi.count("("):
            doi = doi[:-1]
        yield doi.casefold()


def bibliography_from_metadata(document):
    raw = document.get("metadata") or {}
    authors = raw.get("authors")
    return {
        "publication_id": document["publication_id"],
        "title": raw.get("title") or "",
        "year": str(raw.get("year") or ""),
        "DOI": raw.get("DOI") or "",
        "authors": "; ".join(authors) if isinstance(authors, list) else "",
    }


def article_metadata(document):
    texts = [page.get("text", "") for page in document["pages"][:2]]
    raw = document.get("metadata") or {}
    explicit = list(_dois(" ".join(raw.get(field) or "" for field in ("DOI", "subject"))))
    candidates = list(dict.fromkeys([*explicit, *(doi for text in texts for doi in _dois(text))]))[:12]
    page_keys = [_key(text) for text in texts]
    headers = [_key(" ".join(line for line in text.splitlines() if line.strip())[:1800]) for text in texts]
    title = _clean(raw.get("title") or "")
    title_key = _key(title)
    if len(title.split()) < 3 or len(title_key) < 14 or (
        not any(title_key in page for page in page_keys) and _header_score(title_key, headers) < 0.88
    ):
        title = ""

    author = raw.get("author") or ""
    saved_authors = raw.get("authors")
    if not isinstance(saved_authors, list):
        saved_authors = []
    authors = saved_authors
    if not authors and ("," in author or ";" in author or " and " in author):
        authors = [_clean(name) for name in re.split(r",|;|\s+and\s+", author) if name.strip()]

    return {
        "publication_id": document["publication_id"],
        "candidates": candidates,
        "explicit_doi": explicit[0] if explicit else "",
        "page_keys": page_keys,
        "headers": headers,
        "title_keys": [title_key] if title else [],
        "title": title,
        "authors": "; ".join(authors),
        "saved_bibliography": {**bibliography_from_metadata(document), "title": title},
    }


def _crossref(params):
    request = Request(CROSSREF_URL + "?" + urlencode(params), headers={
        "User-Agent": "VINITI-Publication-Metadata/1.0",
    })
    with urlopen(request, timeout=20) as response:
        return json.load(response)["message"]["items"]


def _fetch_works(dois):
    works = {}
    dois = sorted(dois)
    for offset in range(0, len(dois), 16):
        group = dois[offset:offset + 16]
        results = _crossref({
            "filter": ",".join("doi:" + doi for doi in group),
            "select": CROSSREF_FIELDS,
            "rows": len(group),
        })
        for work in results:
            works[work["DOI"].casefold()] = work
    return works


def _title_score(title, article):
    key = _key(_clean(title))
    if len(key) < 14:
        return 0
    if any(key in page for page in article["page_keys"]):
        return 1
    score = max(
        (SequenceMatcher(None, key, other, autojunk=False).ratio() for other in article["title_keys"]
         if len(key) * 0.7 <= len(other) <= len(key) * 1.3),
        default=0,
    )
    return score if score >= 0.88 else max(score, _header_score(key, article["headers"]))


def _choose_work(article, works):
    matches = []
    for doi in article["candidates"]:
        work = works.get(doi)
        if work:
            title = next(iter(work.get("title", [])), "")
            score = _title_score(title, article)
            if score >= 0.88:
                matches.append((score, work))
    if matches:
        return max(matches, key=lambda match: match[0])[1]
    if not article["title"]:
        return None

    key = _key(article["title"])
    works_by_title = _crossref({
        "query.title": article["title"],
        "select": CROSSREF_FIELDS,
        "rows": 4,
    })
    same_title = [
        work for work in works_by_title
        if _key(_clean(next(iter(work.get("title", [])), ""))) == key
    ]
    return same_title[0] if len(same_title) == 1 else None


def _year(work):
    for field in ("published-print", "published", "issued", "published-online"):
        parts = work.get(field, {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return ""


def _authors(work):
    names = []
    for author in work.get("author", []):
        name = author.get("name") or " ".join(
            part for part in (author.get("given"), author.get("family"), author.get("suffix")) if part
        )
        if name:
            names.append(_clean(name))
    return "; ".join(names)


def read_publication_rows(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return {row["publication_id"]: row for row in csv.DictReader(stream)}
    except FileNotFoundError:
        return {}


def apply_publication_metadata(document, row):
    """Add bibliographic fields when the row matches the title in the article."""
    if not row or row["publication_id"] != document["publication_id"]:
        return False
    title = row.get("title")
    if title and _title_score(title, article_metadata(document)) < 0.88:
        return False

    current = document.get("metadata") or {}
    metadata = {key: value for key, value in current.items() if value not in ("", None)}
    if title:
        metadata["title"] = title
    if row.get("DOI"):
        metadata["DOI"] = row["DOI"]
    year = str(row.get("year") or "")
    if re.fullmatch(r"[0-9]{4}", year):
        metadata["year"] = int(year)
    if row.get("authors"):
        names = [name.strip() for name in row["authors"].split(";") if name.strip()]
        if names:
            metadata["authors"] = names
            metadata["author"] = "; ".join(names)

    if metadata == current:
        return False
    document["metadata"] = metadata
    return True


def resolve_publications(articles, previous, *, lookup=True):
    """Resolve the bibliography for prepared articles using saved data and Crossref."""
    verified = {}
    for article in articles:
        publication_id = article["publication_id"]
        stored = previous.get(publication_id, {})
        saved = article["saved_bibliography"]
        if all(saved[field] for field in BIBLIOGRAPHIC_FIELDS):
            verified[publication_id] = saved
        elif _title_score(stored.get("title", ""), article) >= 0.88:
            verified[publication_id] = stored
    pending = {article["publication_id"] for article in articles
               if not all(verified.get(article["publication_id"], {}).get(field) for field in BIBLIOGRAPHIC_FIELDS)}
    dois = {doi for article in articles if article["publication_id"] in pending for doi in article["candidates"]}
    works = _fetch_works(dois) if lookup else {}

    updated = {}
    for article in articles:
        publication_id = article["publication_id"]
        stored = previous.get(publication_id, {})
        known = verified.get(publication_id, {})
        work = (_choose_work(article, works) or {}) if lookup and publication_id in pending else {}
        updated[publication_id] = {
            "publication_id": publication_id,
            "title": _clean(next(iter(work.get("title", [])), "")) or known.get("title") or article["title"],
            "year": _year(work) or known.get("year", ""),
            "DOI": work.get("DOI") or known.get("DOI") or article["explicit_doi"],
            "authors": _authors(work) or known.get("authors") or article["authors"],
            "in_scope": stored.get("in_scope", ""),
            "exclusion_reason": stored.get("exclusion_reason", ""),
        }
    return updated


def missing_bibliography(document):
    row = bibliography_from_metadata(document)
    return [field for field in BIBLIOGRAPHIC_FIELDS if not row[field]]


def write_article_json(path, document):
    temporary = NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False)
    staging_path = Path(temporary.name)
    try:
        with temporary as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        staging_path.replace(path)
    finally:
        staging_path.unlink(missing_ok=True)
