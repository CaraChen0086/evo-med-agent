# URL -> text -> structured chunks -> embeddings -> Chroma server
# query -> query embedding -> retrieve -> EvidenceItem

from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

from .config import VerifierConfig
from .schemas import EvidenceItem


# -----------------------------
# Internal structured chunk data (web)
# -----------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    topic: str
    section: Optional[str]
    chunk_index: int
    text: str               # clean text for storage / LLM consumption
    embedding_text: str     # enriched text for embedding only


# -----------------------------
# PMC-specific internal data
# -----------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


@dataclass
class PMCArticleRecord:
    pmcid: str
    title: str
    journal: str
    pub_year: str
    abstract_parts: List[Tuple[str, str]]
    body_sections: List[Tuple[str, str]]
    matched_queries: List[str]


@dataclass
class PMCChunkRecord:
    chunk_id: str
    document: str
    embedding_text: str
    metadata: Dict[str, object]


class PMCClient:
    def __init__(
        self,
        email: str,
        tool: str = "rag_verifier_pmc_builder",
        api_key: Optional[str] = None,
        timeout: int = 30,
        sleep_seconds: float = 0.34,
    ) -> None:
        self.email = email
        self.tool = tool
        self.api_key = api_key
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"{tool}/1.0 ({email})",
                "Accept": "application/xml,text/xml,application/json,*/*;q=0.8",
            }
        )

    def _base_params(self) -> Dict[str, str]:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get(self, endpoint: str, params: Dict[str, object]) -> requests.Response:
        merged = {**self._base_params(), **params}
        resp = self.session.get(
            f"{EUTILS_BASE}{endpoint}",
            params=merged,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        time.sleep(self.sleep_seconds)
        return resp

    def search_pmc(self, query: str, retmax: int = 100, sort: str = "relevance") -> List[str]:
        resp = self._get(
            "esearch.fcgi",
            {
                "db": "pmc",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": sort,
            },
        )
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_pmc_xml(self, pmc_numeric_id: str) -> str:
        resp = self._get(
            "efetch.fcgi",
            {
                "db": "pmc",
                "id": pmc_numeric_id,
                "retmode": "xml",
            },
        )
        return resp.text


def pmc_normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pmc_clean_text(text: str) -> str:
    text = pmc_normalize_text(text)
    text = re.sub(r"\[(?:\d+(?:[-,]\d+)*)\]", "", text)  # citation brackets
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def pmc_iter_text(elem: ET.Element) -> str:
    return "".join(elem.itertext())


def pmc_dedupe_pairs(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out: List[Tuple[str, str]] = []
    for a, b in items:
        key = (a.lower(), b.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def parse_pmc_article(xml_text: str, matched_queries: List[str]) -> Optional[PMCArticleRecord]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    article = root.find(".//article")
    if article is None:
        article = root

    title = ""
    title_node = article.find(".//front//article-title")
    if title_node is not None:
        title = pmc_clean_text(pmc_iter_text(title_node))

    journal = ""
    journal_node = article.find(".//front//journal-title")
    if journal_node is not None:
        journal = pmc_clean_text(pmc_iter_text(journal_node))

    pub_year = ""
    year_node = article.find(".//front//pub-date/year")
    if year_node is not None and year_node.text:
        pub_year = year_node.text.strip()

    pmcid = ""
    for aid in article.findall(".//article-id"):
        pub_id_type = (aid.attrib.get("pub-id-type") or "").lower()
        val = pmc_clean_text(pmc_iter_text(aid))

        if not val:
            continue

        if pub_id_type in {"pmc", "pmcid", "pmcid-ver"}:
            pmcid = val
            break

        if not pmcid and val.upper().startswith("PMC"):
            pmcid = val

    abstract_parts: List[Tuple[str, str]] = []
    for abstract in article.findall(".//front//abstract") or article.findall(".//abstract"):
        sec_title_node = abstract.find("./title")
        sec_name = pmc_clean_text(pmc_iter_text(sec_title_node)) if sec_title_node is not None else "Abstract"
        for p in abstract.findall(".//p"):
            text = pmc_clean_text(pmc_iter_text(p))
            if text:
                abstract_parts.append((sec_name, text))

    body_sections: List[Tuple[str, str]] = []
    body = article.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            sec_title_node = sec.find("./title")
            sec_name = pmc_clean_text(pmc_iter_text(sec_title_node)) if sec_title_node is not None else "Body"
            for p in sec.findall("./p"):
                text = pmc_clean_text(pmc_iter_text(p))
                if text:
                    body_sections.append((sec_name, text))

        if not body_sections:
            for p in body.findall(".//p"):
                text = pmc_clean_text(pmc_iter_text(p))
                if text:
                    body_sections.append(("Body", text))

    if not pmcid:
        pmcid = "UNKNOWN"

    return PMCArticleRecord(
        pmcid=pmcid,
        title=title or pmcid,
        journal=journal,
        pub_year=pub_year,
        abstract_parts=pmc_dedupe_pairs(abstract_parts),
        body_sections=pmc_dedupe_pairs(body_sections),
        matched_queries=matched_queries,
    )


def chunk_pmc_article(
    article: PMCArticleRecord,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 250,
) -> List[PMCChunkRecord]:
    blocks: List[Tuple[str, str]] = []
    blocks.extend([(f"Abstract: {sec}", txt) for sec, txt in article.abstract_parts])
    blocks.extend(article.body_sections)

    chunks: List[PMCChunkRecord] = []
    current = ""
    current_section = "Mixed"
    chunk_idx = 0

    def finalize(section_name: str, body: str) -> None:
        nonlocal chunk_idx
        body = body.strip()
        if len(body) < min_chunk_chars:
            return

        header = [f"[Title: {article.title}]"]
        header.append(f"[Section: {section_name}]")
        if article.journal:
            header.append(f"[Journal: {article.journal}]")
        if article.pub_year:
            header.append(f"[Year: {article.pub_year}]")
        if article.matched_queries:
            header.append(f"[MatchedQueries: {' | '.join(article.matched_queries[:5])}]")

        stored = "\n".join(header) + "\n" + body
        embedding_text = (
            f"[SourceType: PMC Full Text]\n"
            f"[PMCID: {article.pmcid}]\n"
            + stored
        )

        chunk_id = hashlib.md5(
            f"{article.pmcid}::{section_name}::{chunk_idx}::{body[:120]}".encode("utf-8")
        ).hexdigest()

        chunks.append(
            PMCChunkRecord(
                chunk_id=chunk_id,
                document=stored,
                embedding_text=embedding_text,
                metadata={
                    "source_type": "pmc",
                    "source": f"https://pmc.ncbi.nlm.nih.gov/articles/{article.pmcid}/",
                    "pmcid": article.pmcid,
                    "title": article.title,
                    "section": section_name,
                    "journal": article.journal,
                    "pub_year": article.pub_year,
                    "chunk_index": chunk_idx,
                    "matched_queries": " | ".join(article.matched_queries[:10]),
                    "length": len(body),
                },
            )
        )
        chunk_idx += 1

    for section_name, paragraph in blocks:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            current_section = section_name
            continue

        if current:
            finalize(current_section, current)
            tail = current[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
            current_section = section_name
        else:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end].strip()
                if len(piece) >= min_chunk_chars:
                    finalize(section_name, piece)
                if end == len(paragraph):
                    break
                start = max(end - chunk_overlap, start + 1)
            current = ""

    if current:
        finalize(current_section, current)

    max_chunks_per_article = 80
    if len(chunks) > max_chunks_per_article:
        chunks = chunks[:max_chunks_per_article]
    return chunks


def collect_candidate_articles(
    client: PMCClient,
    queries: List[str],
    max_per_query: int,
    max_total_articles: int,
    sort: str,
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}

    for query in queries:
        ids = client.search_pmc(query=query, retmax=max_per_query, sort=sort)
        for pmc_id in ids:
            if pmc_id not in merged:
                merged[pmc_id] = []
            merged[pmc_id].append(query)
            if len(merged) >= max_total_articles:
                return merged

    return merged


def batched(items: List[object], batch_size: int) -> Iterable[List[object]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class KnowledgeBase:
    """
    Goals of this version:
    1. better cleaning for webpage text
    2. topic/title extraction
    3. heading-aware chunking
    4. discard overly tiny chunks
    5. separate embedding_text from stored text
    """

    def __init__(self, config: VerifierConfig):
        self.config = config

        if chromadb is None or SentenceTransformer is None:
            raise ImportError(
                "Missing dependencies. Install: chromadb sentence-transformers beautifulsoup4 requests"
            )

        self.embedder = SentenceTransformer(
            config.embedding_model,
            trust_remote_code=True,
        )

        self.client = chromadb.HttpClient(
            host=config.chroma_host,
            port=config.chroma_port,
        )

        # main retrieval collection (can hold web + PMC chunks)
        self.collection_name = getattr(config, "collection_name", "rag_verifier_kb")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        # safe fallbacks in case your VerifierConfig doesn't yet define them
        self.chunk_size = getattr(config, "chunk_size", 1000)
        self.chunk_overlap = getattr(config, "chunk_overlap", 150)
        self.min_chunk_len = getattr(config, "min_chunk_len", 200)
        self.max_title_len = getattr(config, "max_title_len", 120)
        self.top_k = getattr(config, "retrieval_top_k", getattr(config, "top_k", 5))

    # -----------------------------
    # Collection lifecycle
    # -----------------------------
    def reset_collection(self) -> None:
        """
        Drop and recreate the active collection.
        Used when forcing a fresh scenario-specific PMC KB.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"[KB] Failed to delete collection {self.collection_name}: {e}")
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    # -----------------------------
    # Fetching + cleaning
    # -----------------------------
    def fetch_url_html(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[KB] Error fetching {url}: {e}")
            return ""

    def fetch_url_text(self, url: str) -> str:
        """
        Main public fetch API.
        Kept for compatibility, but now delegates to the structured cleaner.
        """
        html_text = self.fetch_url_html(url)
        if not html_text:
            return ""
        text, _ = self.extract_structured_text(url, html_text)
        return text

    def extract_structured_text(self, url: str, html_text: str) -> Tuple[str, str]:
        """
        Returns:
            cleaned_text: normalized text with paragraph/heading structure
            topic: extracted title/topic
        """
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove obvious junk containers
        for tag in soup([
            "script", "style", "nav", "footer", "header", "aside",
            "form", "button", "noscript", "svg", "canvas"
        ]):
            tag.decompose()

        # Try a focused main-content extraction first
        main_node = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.body
            or soup
        )

        topic = self.extract_topic(url, soup, main_node)

        # Walk tags in order to preserve basic document structure
        blocks: List[str] = []
        allowed_tags = {
            "h1", "h2", "h3", "h4",
            "p", "li", "blockquote",
            "dt", "dd"
        }

        for el in main_node.find_all(allowed_tags):
            text = self.normalize_text(el.get_text(" ", strip=True))
            if not text:
                continue

            if self.is_noise_line(text):
                continue

            tag_name = getattr(el, "name", "")

            # Heading marker so downstream chunker can detect section boundary
            if tag_name in {"h1", "h2", "h3", "h4"}:
                if self.is_valid_heading(text):
                    blocks.append(f"## {text}")
                continue

            # list items: keep bullet signal
            if tag_name in {"li", "dt", "dd"}:
                blocks.append(f"- {text}")
            else:
                blocks.append(text)

        cleaned_lines = self.postprocess_blocks(blocks)
        cleaned_text = "\n\n".join(cleaned_lines).strip()

        return cleaned_text, topic

    # -----------------------------
    # Topic / title extraction
    # -----------------------------
    def extract_topic(self, url: str, soup: BeautifulSoup, main_node: BeautifulSoup) -> str:
        """
        Priority:
        1. OpenGraph/twitter title
        2. <title>
        3. first high-quality heading in main content
        4. URL slug fallback
        """
        meta_candidates = [
            ("meta", {"property": "og:title"}),
            ("meta", {"name": "twitter:title"}),
            ("meta", {"name": "title"}),
        ]
        for tag_name, attrs in meta_candidates:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content"):
                title = self.normalize_text(tag["content"])
                if self.is_valid_title(title):
                    return title[: self.max_title_len]

        if soup.title and soup.title.string:
            title = self.normalize_text(soup.title.string)
            if self.is_valid_title(title):
                return title[: self.max_title_len]

        for h in main_node.find_all(["h1", "h2"]):
            text = self.normalize_text(h.get_text(" ", strip=True))
            if self.is_valid_title(text):
                return text[: self.max_title_len]

        tail = url.rstrip("/").split("/")[-1]
        tail = re.sub(r"[-_]+", " ", tail)
        tail = re.sub(r"\.[a-zA-Z0-9]+$", "", tail)
        tail = self.normalize_text(tail)

        if tail:
            return tail.title()[: self.max_title_len]

        return "Unknown Topic"

    # -----------------------------
    # Text normalization helpers
    # -----------------------------
    def normalize_text(self, text: str) -> str:
        text = html.unescape(text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def is_noise_line(self, line: str) -> bool:
        low = line.lower().strip()

        noise_keywords = [
            "skip to content",
            "skip navigation",
            "official website",
            "privacy policy",
            "terms of use",
            "terms & conditions",
            "cookie policy",
            "all rights reserved",
            "copyright",
            "sign in",
            "log in",
            "subscribe",
            "advertisement",
            "advertising",
            "share this page",
            "follow us",
            "click here",
            "print this page",
            "back to top",
        ]
        if any(k in low for k in noise_keywords):
            return True

        # too short and not likely informative
        if len(line) < 3:
            return True

        # repeated UI-ish fragments
        if re.fullmatch(r"[\W_]+", line):
            return True

        return False

    def is_valid_title(self, text: str) -> bool:
        if not text:
            return False
        if len(text) < 6 or len(text) > 160:
            return False
        if self.is_noise_line(text):
            return False
        return True

    def is_valid_heading(self, text: str) -> bool:
        if not text:
            return False
        if len(text) > 120:
            return False
        if self.is_noise_line(text):
            return False

        # headings are often short; unlike generic lines, we allow them
        return True

    def postprocess_blocks(self, blocks: List[str]) -> List[str]:
        """
        Deduplicate local repetitions, keep structure, discard weak fragments.
        """
        cleaned: List[str] = []
        seen_recent = set()

        for block in blocks:
            block = self.normalize_text(block)
            if not block:
                continue

            # normalize heading marker spacing
            if block.startswith("##"):
                block = "## " + block[2:].strip()

            key = block.lower()
            if key in seen_recent:
                continue

            cleaned.append(block)
            seen_recent.add(key)

            # keep local dedupe bounded
            if len(seen_recent) > 200:
                seen_recent = set(list(seen_recent)[-100:])

        return cleaned

    # -----------------------------
    # Chunking
    # -----------------------------
    def chunk_text(
        self,
        text: str,
        topic: Optional[str] = None,
        min_chunk_len: Optional[int] = None,
    ) -> List[ChunkRecord]:
        """
        Heading-aware chunking:
        - treat "## Heading" as a section boundary
        - keep heading with following content
        - split long sections with overlap
        - discard tiny chunks
        """
        if not text:
            return []

        min_chunk_len = min_chunk_len or self.min_chunk_len

        sections = self.split_into_sections(text)
        chunk_records: List[ChunkRecord] = []
        chunk_index = 0

        for section_heading, section_body in sections:
            section_chunks = self.chunk_section(
                section_heading=section_heading,
                section_body=section_body,
                topic=topic,
                min_chunk_len=min_chunk_len,
            )
            for rec in section_chunks:
                rec.chunk_index = chunk_index
                chunk_records.append(rec)
                chunk_index += 1

        # merge too-small trailing chunks if needed
        chunk_records = self.merge_tiny_chunks(chunk_records, min_chunk_len=min_chunk_len)

        return chunk_records

    def split_into_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """
        Input text is expected to contain heading markers like:
            ## Contraindications
        """
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        sections: List[Tuple[Optional[str], List[str]]] = []

        current_heading: Optional[str] = None
        current_body: List[str] = []

        for part in parts:
            if part.startswith("## "):
                # flush current section
                if current_heading is not None or current_body:
                    sections.append((current_heading, current_body))
                current_heading = part[3:].strip()
                current_body = []
            else:
                current_body.append(part)

        if current_heading is not None or current_body:
            sections.append((current_heading, current_body))

        # convert body list to text
        final_sections: List[Tuple[Optional[str], str]] = []
        for heading, body_list in sections:
            body = "\n\n".join(body_list).strip()
            if heading and body:
                final_sections.append((heading, body))
            elif heading and not body:
                # keep empty heading out
                continue
            elif body:
                final_sections.append((None, body))

        return final_sections

    def chunk_section(
        self,
        section_heading: Optional[str],
        section_body: str,
        topic: Optional[str],
        min_chunk_len: int,
    ) -> List[ChunkRecord]:
        """
        Chunk a single section while preserving heading.
        """
        if not section_body:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", section_body) if p.strip()]
        size = self.chunk_size
        overlap = self.chunk_overlap

        results: List[ChunkRecord] = []
        current = ""

        def finalize_chunk(chunk_text: str) -> Optional[ChunkRecord]:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < min_chunk_len:
                return None

            # stored text: clean for final evidence display
            stored_text = self.build_stored_text(
                topic=topic,
                section=section_heading,
                body=chunk_text,
            )

            # embedding text: stronger retrieval signal
            embedding_text = self.build_embedding_text(
                topic=topic,
                section=section_heading,
                body=chunk_text,
            )

            return ChunkRecord(
                chunk_id="",  # set later in build_from_urls
                source="",
                topic=topic or "Unknown Topic",
                section=section_heading,
                chunk_index=-1,
                text=stored_text,
                embedding_text=embedding_text,
            )

        for para in paragraphs:
            # if one paragraph is too large, split internally
            if len(para) > size:
                if current.strip():
                    rec = finalize_chunk(current)
                    if rec:
                        results.append(rec)
                    current = ""

                for piece in self.split_long_text(para, size=size, overlap=overlap):
                    rec = finalize_chunk(piece)
                    if rec:
                        results.append(rec)
                continue

            candidate = f"{current}\n\n{para}".strip() if current else para

            if len(candidate) <= size:
                current = candidate
            else:
                rec = finalize_chunk(current)
                if rec:
                    results.append(rec)

                # overlap by tail chars from previous current
                if current and overlap > 0:
                    tail = current[-overlap:].strip()
                    current = f"{tail}\n\n{para}".strip()
                else:
                    current = para

        if current.strip():
            rec = finalize_chunk(current)
            if rec:
                results.append(rec)

        return results

    def split_long_text(self, text: str, size: int, overlap: int) -> List[str]:
        """
        Soft split:
        - prefer sentence-ish boundaries
        - fallback to char window
        """
        if len(text) <= size:
            return [text]

        pieces: List[str] = []
        start = 0

        while start < len(text):
            max_end = min(start + size, len(text))
            if max_end == len(text):
                piece = text[start:max_end].strip()
                if piece:
                    pieces.append(piece)
                break

            window = text[start:max_end]

            # Prefer boundary near the end
            candidates = [
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("; "),
                window.rfind(": "),
                window.rfind(", "),
            ]
            cut = max(candidates)

            if cut < int(size * 0.5):
                cut = len(window)

            piece = window[:cut].strip()
            if piece:
                pieces.append(piece)

            if cut == 0:
                start = max_end
            else:
                start = start + cut - overlap
                if start < 0:
                    start = 0

        return pieces

    def merge_tiny_chunks(
        self,
        chunks: List[ChunkRecord],
        min_chunk_len: int,
    ) -> List[ChunkRecord]:
        """
        Merge tiny trailing chunks with previous chunk if same topic/section.
        """
        if not chunks:
            return []

        merged: List[ChunkRecord] = []
        for rec in chunks:
            body_len = len(rec.text)
            if (
                merged
                and body_len < min_chunk_len
                and merged[-1].topic == rec.topic
                and merged[-1].section == rec.section
            ):
                prev = merged[-1]
                merged_body = f"{prev.text}\n\n{rec.text}".strip()
                merged_embed = f"{prev.embedding_text}\n\n{rec.embedding_text}".strip()

                merged[-1] = ChunkRecord(
                    chunk_id=prev.chunk_id,
                    source=prev.source,
                    topic=prev.topic,
                    section=prev.section,
                    chunk_index=prev.chunk_index,
                    text=merged_body,
                    embedding_text=merged_embed,
                )
            else:
                merged.append(rec)

        # reindex
        for i, rec in enumerate(merged):
            rec.chunk_index = i

        return merged

    def build_stored_text(
        self,
        topic: Optional[str],
        section: Optional[str],
        body: str,
    ) -> str:
        """
        This is the text returned to the model at query time.
        Keep it informative but not overly polluted.
        """
        prefix_lines = []
        if topic:
            prefix_lines.append(f"[Topic: {topic}]")
        if section:
            prefix_lines.append(f"[Section: {section}]")

        prefix = "\n".join(prefix_lines).strip()
        if prefix:
            return f"{prefix}\n{body}".strip()
        return body.strip()

    def build_embedding_text(
        self,
        topic: Optional[str],
        section: Optional[str],
        body: str,
    ) -> str:
        """
        This is the text used only for embeddings.
        Make the retrieval bias stronger here.
        """
        prefix_lines = []
        if topic:
            prefix_lines.append(f"[Topic: {topic}]")
        if section:
            prefix_lines.append(f"[Section: {section}]")
        prefix_lines.append("[SourceType: Web Medical Knowledge]")

        prefix = "\n".join(prefix_lines).strip()
        return f"{prefix}\n{body}".strip()

    # -----------------------------
    # Build KB
    # -----------------------------
    def build_from_urls(self, urls: List[str] | None = None) -> int:
        urls = urls or self.config.url_list

        ids: List[str] = []
        stored_docs: List[str] = []
        embedding_docs: List[str] = []
        metas: List[dict] = []

        total_chunks = 0

        for url in urls:
            html_text = self.fetch_url_html(url)
            if not html_text:
                continue

            raw_text, topic = self.extract_structured_text(url, html_text)
            if not raw_text:
                continue

            chunk_records = self.chunk_text(
                raw_text,
                topic=topic,
                min_chunk_len=self.min_chunk_len,
            )

            for i, rec in enumerate(chunk_records):
                chunk_id = hashlib.md5(f"{url}::{topic}::{i}".encode()).hexdigest()
                rec.chunk_id = chunk_id
                rec.source = url
                rec.chunk_index = i

                ids.append(chunk_id)
                stored_docs.append(rec.text)
                embedding_docs.append(rec.embedding_text)
                metas.append(
                    {
                        "source": url,
                        "source_type": "web",
                        "topic": rec.topic,
                        "section": rec.section or "",
                        "chunk_index": i,
                        "raw_length": len(rec.text),
                    }
                )

            total_chunks += len(chunk_records)

        if not stored_docs:
            return 0

        embeddings = self.embedder.encode(
            embedding_docs,
            normalize_embeddings=True,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=stored_docs,
            embeddings=embeddings,
            metadatas=metas,
        )

        return total_chunks

    def build_from_pmc_queries(self, queries: List[str]) -> int:
        """
        Build or extend a scenario-focused PMC KB inside the current collection.
        """
        if not queries:
            print("[KB][PMC] No queries provided, skipping PMC build.")
            return 0

        email = getattr(self.config, "pmc_email", "") or ""
        if not email or email == "your_email@example.com":
            print("[KB][PMC] pmc_email is not configured, skipping PMC build.")
            return 0

        api_key = getattr(self.config, "pmc_api_key", None)
        max_per_query = getattr(self.config, "pmc_max_per_query", 10)
        max_total_articles = getattr(self.config, "pmc_max_total_articles", 50)

        client = PMCClient(
            email=email,
            tool="rag_verifier_pmc_builder",
            api_key=api_key,
            sleep_seconds=0.11 if api_key else 0.34,
        )

        # Selection
        print("[KB][PMC] Queries to search:")
        for q in queries:
            print(" -", q)

        candidate_map = collect_candidate_articles(
            client=client,
            queries=queries,
            max_per_query=max_per_query,
            max_total_articles=max_total_articles,
            sort="relevance",
        )

        print(f"[KB][PMC] Unique candidate articles selected: {len(candidate_map)}")

        # Fetch + parse
        articles: List[PMCArticleRecord] = []
        for idx, (numeric_pmc_id, matched_queries) in enumerate(candidate_map.items(), start=1):
            try:
                xml_text = client.fetch_pmc_xml(numeric_pmc_id)
                article = parse_pmc_article(xml_text, matched_queries=matched_queries)
                if not article:
                    print(f"[KB][PMC] Skip {numeric_pmc_id}: parse failed")
                    continue
                if not article.abstract_parts and not article.body_sections:
                    print(f"[KB][PMC] Skip {article.pmcid}: no usable text")
                    continue

                title_low = (article.title or "").lower()
                if "abstracts from" in title_low:
                    print(f"[KB][PMC] Skip {article.pmcid}: conference abstract collection")
                    continue
                if "annual meeting" in title_low:
                    print(f"[KB][PMC] Skip {article.pmcid}: annual meeting abstract collection")
                    continue

                total_body_paras = len(article.body_sections)
                if total_body_paras > 300:
                    print(f"[KB][PMC] Skip {article.pmcid}: too many body paragraphs ({total_body_paras})")
                    continue

                articles.append(article)
                print(
                    f"[KB][PMC] {idx}/{len(candidate_map)} "
                    f"{article.pmcid} | year={article.pub_year or 'NA'} | title={article.title[:80]}"
                )
            except Exception as e:
                print(f"[KB][PMC] Skip {numeric_pmc_id}: {e}")

        print(f"[KB][PMC] Parsed articles kept: {len(articles)}")

        # Chunk
        all_chunks: List[PMCChunkRecord] = []
        for article in articles:
            chunks = chunk_pmc_article(
                article=article,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_chars=self.min_chunk_len,
            )
            all_chunks.extend(chunks)

        print(f"[KB][PMC] Total chunks produced: {len(all_chunks)}")
        if not all_chunks:
            print("[KB][PMC] No chunks to upsert.")
            return 0

        # Upsert into existing collection using shared embedder
        batch_size = 32
        total = 0
        for batch in batched(all_chunks, batch_size):
            ids = [c.chunk_id for c in batch]
            docs = [c.document for c in batch]
            embed_texts = [c.embedding_text for c in batch]
            metas = [c.metadata for c in batch]

            embeddings = self.embedder.encode(
                embed_texts,
                normalize_embeddings=True,
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metas,
            )
            total += len(batch)
            first = metas[0] if metas else {}
            print(
                f"[KB][PMC] Upserted batch: {len(batch)} chunks "
                f"(example pmcid={first.get('pmcid', 'NA')}, section={first.get('section', 'NA')})"
            )

        print(f"[KB][PMC] Upsert finished, total chunks upserted: {total}")
        return total

# -----------------------------
# Query
# -----------------------------
    def query(
        self,
        query_texts: List[str],
        n_results: int | None = None,
    ) -> List[EvidenceItem]:

        n_results = n_results or self.top_k

        if not query_texts:
            return []

        # 🔥 只在 BGE / Qwen embedding 时加 instruction
        if "bge" in self.config.embedding_model.lower() or "qwen" in self.config.embedding_model.lower():
            instruction = "Represent this query for retrieving relevant medical documents: "
            processed_queries = [instruction + q for q in query_texts]
        else:
            processed_queries = query_texts

        # embedding
        q_emb = self.embedder.encode(
            processed_queries,
            normalize_embeddings=True,
        ).tolist()

        # retrieval
        result = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
        )

        evidence: Dict[str, EvidenceItem] = {}

        for q_idx in range(len(query_texts)):
            ids = result.get("ids", [[]])[q_idx]
            docs = result.get("documents", [[]])[q_idx]
            metas = result.get("metadatas", [[]])[q_idx]
            distances = result.get("distances", [[]])[q_idx]

            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                if doc_id not in evidence:
                    topic = meta.get("topic", "")
                    section = meta.get("section", "")

                    display_text = doc

                    # 🔥 确保 topic header 存在
                    if topic and not display_text.startswith("[Topic:"):
                        header = f"[Topic: {topic}]"
                        if section:
                            header += f"\n[Section: {section}]"
                        display_text = f"{header}\n{doc}"

                    evidence[doc_id] = EvidenceItem(
                        source=meta.get("source", "unknown"),
                        chunk_id=doc_id,
                        text=display_text,
                        distance=dist,
                        source_type=meta.get("source_type", "web"),
                    )

        return list(evidence.values())

    def build_from_acr_items(self, acr_items: List[Dict]) -> int:
        """
        Insert ACR-derived structured items into the KB collection.
        Reuses existing embedder and collection.
        """
        if not acr_items:
            return 0

        ids: List[str] = []
        stored_docs: List[str] = []
        embedding_docs: List[str] = []
        metas: List[Dict] = []

        total_chunks = 0

        for item in acr_items:
            chunks = self.acr_item_to_chunks(item)
            for i, chunk_text in enumerate(chunks):
                chunk_id = hashlib.md5(f"acr::{item.get('url', 'unknown')}::{i}".encode()).hexdigest()

                # Stored text: clean for LLM consumption
                stored_text = chunk_text

                # Embedding text: add retrieval signals
                embedding_text = f"[SourceType: ACR Rating Table]\n{chunk_text}"

                ids.append(chunk_id)
                stored_docs.append(stored_text)
                embedding_docs.append(embedding_text)
                metas.append({
                    "source": item.get("url", "unknown"),
                    "source_type": "acr",
                    "topic": item.get("topic_name", "Unknown"),
                    "scenario_text": item.get("scenario_text", ""),
                    "variant_text": item.get("variant_text", ""),
                    "procedure": "",  # Will be set per recommendation
                    "appropriateness": "",  # Will be set per recommendation
                    "section": "rating_table",
                    "chunk_index": i,
                    "raw_length": len(chunk_text),
                })

            total_chunks += len(chunks)

        if not stored_docs:
            return 0

        embeddings = self.embedder.encode(
            embedding_docs,
            normalize_embeddings=True,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=stored_docs,
            embeddings=embeddings,
            metadatas=metas,
        )

        print(f"[KB][ACR] Upserted {total_chunks} ACR chunks")
        return total_chunks

    def acr_item_to_chunks(self, item: Dict) -> List[str]:
        """
        Convert ACR item to chunk texts.
        Each recommendation becomes a short chunk.
        """
        chunks = []
        base_info = f"[Topic: {item.get('topic_name', 'Unknown')}]\n[Panel: {item.get('panel', 'Unknown')}]"

        if item.get('scenario_text'):
            base_info += f"\n[Scenario: {item['scenario_text']}]"

        if item.get('variant_text'):
            base_info += f"\n[Variant: {item['variant_text']}]"

        for rec in item.get('recommendations', []):
            chunk = f"{base_info}\nProcedure: {rec['procedure']}\nAppropriateness: {rec['appropriateness']}"
            chunks.append(chunk)

        return chunks
