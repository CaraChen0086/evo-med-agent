"""
Scenario-focused PMC KB builder for Chroma.

Recommended workflow for clinical verifier:
1) Start from a scenario or a small set of hand-written topic queries
2) Search PMC with multiple focused queries
3) Deduplicate articles across queries
4) Fetch full-text XML
5) Parse abstract/body sections
6) Chunk and embed
7) Upsert into a Chroma collection

Why this design:
- Better than building "all of PMC" for a narrow clinical verifier
- Keeps the KB relevant and small
- Easy to extend incrementally as new scenarios arrive

Example:
python build_pmc_kb_v2.py \
  --queries_file queries.txt \
  --email your_email@example.com \
  --collection_name pmc_kb_demo \
  --max_per_query 50 \
  --max_total_articles 200 \
  --embedding_model Qwen/Qwen3-Embedding-8B
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import chromadb
import requests
from sentence_transformers import SentenceTransformer


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class ArticleRecord:
    pmcid: str
    title: str
    journal: str
    pub_year: str
    abstract_parts: List[Tuple[str, str]]
    body_sections: List[Tuple[str, str]]
    matched_queries: List[str]


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    embedding_text: str
    metadata: Dict[str, object]


# -----------------------------
# NCBI / PMC Client
# -----------------------------
class PMCClient:
    def __init__(
        self,
        email: str,
        tool: str = "clinical_kb_builder",
        api_key: Optional[str] = None,
        timeout: int = 30,
        sleep_seconds: float = 0.34,  # <= 3 req/sec without api_key
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
        """
        Returns numeric PMC IDs from db=pmc search.
        """
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
        """
        Fetches full text XML for a PMC record using db=pmc.
        """
        resp = self._get(
            "efetch.fcgi",
            {
                "db": "pmc",
                "id": pmc_numeric_id,
                "retmode": "xml",
            },
        )
        return resp.text


# -----------------------------
# Text helpers
# -----------------------------
def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\[(?:\d+(?:[-,]\d+)*)\]", "", text)  # citation brackets
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def iter_text(elem: ET.Element) -> str:
    return "".join(elem.itertext())


# -----------------------------
# XML parser
# -----------------------------
def parse_pmc_article(xml_text: str, matched_queries: List[str]) -> Optional[ArticleRecord]:
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
        title = clean_text(iter_text(title_node))

    journal = ""
    journal_node = article.find(".//front//journal-title")
    if journal_node is not None:
        journal = clean_text(iter_text(journal_node))

    pub_year = ""
    year_node = article.find(".//front//pub-date/year")
    if year_node is not None and year_node.text:
        pub_year = year_node.text.strip()

    pmcid = ""
    for aid in article.findall(".//article-id"):
        pub_id_type = (aid.attrib.get("pub-id-type") or "").lower()
        val = clean_text(iter_text(aid))

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
        sec_name = clean_text(iter_text(sec_title_node)) if sec_title_node is not None else "Abstract"
        for p in abstract.findall(".//p"):
            text = clean_text(iter_text(p))
            if text:
                abstract_parts.append((sec_name, text))

    body_sections: List[Tuple[str, str]] = []
    body = article.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            sec_title_node = sec.find("./title")
            sec_name = clean_text(iter_text(sec_title_node)) if sec_title_node is not None else "Body"
            for p in sec.findall("./p"):
                text = clean_text(iter_text(p))
                if text:
                    body_sections.append((sec_name, text))

        if not body_sections:
            for p in body.findall(".//p"):
                text = clean_text(iter_text(p))
                if text:
                    body_sections.append(("Body", text))

    if not pmcid:
        pmcid = "UNKNOWN"

    return ArticleRecord(
        pmcid=pmcid,
        title=title or pmcid,
        journal=journal,
        pub_year=pub_year,
        abstract_parts=dedupe_pairs(abstract_parts),
        body_sections=dedupe_pairs(body_sections),
        matched_queries=matched_queries,
    )


def dedupe_pairs(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out: List[Tuple[str, str]] = []
    for a, b in items:
        key = (a.lower(), b.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


# -----------------------------
# Chunking
# -----------------------------
def chunk_article(
    article: ArticleRecord,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 250,
) -> List[ChunkRecord]:
    blocks: List[Tuple[str, str]] = []
    blocks.extend([(f"Abstract: {sec}", txt) for sec, txt in article.abstract_parts])
    blocks.extend(article.body_sections)

    chunks: List[ChunkRecord] = []
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
            ChunkRecord(
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


# -----------------------------
# Query generation
# -----------------------------
def load_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = []

    if args.query:
        queries.append(args.query.strip())

    if args.queries_file:
        for line in Path(args.queries_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)

    if args.scenario_json:
        scenario = json.loads(Path(args.scenario_json).read_text())
        queries.extend(generate_queries_from_scenario(scenario))

    queries = [q for q in queries if q]
    deduped = []
    seen = set()
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped


def generate_queries_from_scenario(s: Dict[str, object]) -> List[str]:
    details = s.get("modification_details", {}) or {}
    pred = s.get("predicted_outcome", {}) or {}
    question = str(s.get("question", "")).strip()
    rationale = str(s.get("rationale", "")).strip()
    changed_targets = details.get("changed_targets", []) or []
    description = str(details.get("description", "")).strip()
    raw_name = "WBC" if "raw_WBC" in pred else ""
    base_terms = []

    if question:
        base_terms.append(question)
    if rationale:
        base_terms.append(rationale)
    if description:
        base_terms.append(description)
    if changed_targets:
        base_terms.extend([str(x) for x in changed_targets])

    joined = " ".join(base_terms)

    # Focused query set for this type of clinical counterfactual
    queries = [
        joined,
        "MRSA antibiotic treatment vancomycin postoperative infection review",
        "leukocytosis white blood cell discharge infection readmission review",
        "postoperative empyema abscess management review",
        "hospital discharge infection risk predictor postoperative review",
    ]

    if raw_name == "WBC":
        queries.append("white blood cell elevation infection risk postoperative review")

    return dedupe_preserve_order(queries)


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


# -----------------------------
# Selection strategy
# -----------------------------
def collect_candidate_articles(
    client: PMCClient,
    queries: List[str],
    max_per_query: int,
    max_total_articles: int,
    sort: str,
) -> Dict[str, List[str]]:
    """
    Returns mapping:
        numeric_pmc_id -> [matched queries...]
    Selection rule:
    - Search each focused query independently
    - Keep order from ESearch relevance ranking
    - Merge by first appearance across queries
    - Track which queries matched each article
    """
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


# -----------------------------
# Chroma ingest
# -----------------------------
def batched(items: List[object], batch_size: int) -> Iterable[List[object]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def upsert_chunks(
    chunks: List[ChunkRecord],
    chroma_host: str,
    chroma_port: int,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
) -> None:
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(name=collection_name)
    embedder = SentenceTransformer(embedding_model, trust_remote_code=True)

    for batch in batched(chunks, batch_size):
        ids = [c.chunk_id for c in batch]
        docs = [c.document for c in batch]
        embed_texts = [c.embedding_text for c in batch]
        metas = [c.metadata for c in batch]

        embeddings = embedder.encode(
            embed_texts,
            normalize_embeddings=True,
            batch_size=min(8, len(embed_texts)),
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )


# -----------------------------
# Main builder
# -----------------------------
def build_pmc_kb(args: argparse.Namespace) -> None:
    queries = load_queries(args)
    if not queries:
        raise ValueError("No queries provided. Use --query, --queries_file, or --scenario_json.")

    client = PMCClient(
        email=args.email,
        tool=args.tool,
        api_key=args.api_key,
        sleep_seconds=0.11 if args.api_key else 0.34,
    )

    print("[PMC] Queries to search:")
    for q in queries:
        print(" -", q)

    candidate_map = collect_candidate_articles(
        client=client,
        queries=queries,
        max_per_query=args.max_per_query,
        max_total_articles=args.max_total_articles,
        sort=args.sort,
    )

    print(f"[PMC] Unique candidate articles selected: {len(candidate_map)}")

    articles: List[ArticleRecord] = []
    for idx, (numeric_pmc_id, matched_queries) in enumerate(candidate_map.items(), start=1):
        try:
            xml_text = client.fetch_pmc_xml(numeric_pmc_id)
            article = parse_pmc_article(xml_text, matched_queries=matched_queries)
            if not article:
                print(f"[PMC] Skip {numeric_pmc_id}: parse failed")
                continue
            if not article.abstract_parts and not article.body_sections:
                print(f"[PMC] Skip {article.pmcid}: no usable text")
                continue
            title_low = (article.title or "").lower()

            if "abstracts from" in title_low:
                print(f"[PMC] Skip {article.pmcid}: conference abstract collection")
                continue

            if "annual meeting" in title_low:
                print(f"[PMC] Skip {article.pmcid}: annual meeting abstract collection")
                continue

            total_body_paras = len(article.body_sections)
            if total_body_paras > 300:
                print(f"[PMC] Skip {article.pmcid}: too many body paragraphs ({total_body_paras})")
                continue
            articles.append(article)
            print(
                f"[PMC] {idx}/{len(candidate_map)} "
                f"{article.pmcid} | year={article.pub_year or 'NA'} | title={article.title[:80]}"
            )
        except Exception as e:
            print(f"[PMC] Skip {numeric_pmc_id}: {e}")

    print(f"[PMC] Parsed articles kept: {len(articles)}")

    all_chunks: List[ChunkRecord] = []
    for article in articles:
        chunks = chunk_article(
            article=article,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_chars=args.min_chunk_chars,
        )
        all_chunks.extend(chunks)

    print(f"[PMC] Total chunks produced: {len(all_chunks)}")
    if not all_chunks:
        print("[PMC] No chunks to upsert.")
        return

    upsert_chunks(
        chunks=all_chunks,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    if args.manifest_out:
        manifest = {
            "queries": queries,
            "selected_article_count": len(articles),
            "chunk_count": len(all_chunks),
            "collection_name": args.collection_name,
            "articles": [
                {
                    "pmcid": a.pmcid,
                    "title": a.title,
                    "journal": a.journal,
                    "pub_year": a.pub_year,
                    "matched_queries": a.matched_queries,
                }
                for a in articles
            ],
        }
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2))
        print(f"[PMC] Manifest saved to {args.manifest_out}")

    print(f"[PMC] Done. Collection: {args.collection_name}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scenario-focused PMC KB in Chroma")

    # query inputs
    parser.add_argument("--query", type=str, default=None, help="Single PMC query")
    parser.add_argument("--queries_file", type=str, default=None, help="Text file with one query per line")
    parser.add_argument("--scenario_json", type=str, default=None, help="Scenario JSON file for auto query generation")

    # ncbi / pmc
    parser.add_argument("--email", type=str, required=True, help="Email for NCBI E-utilities")
    parser.add_argument("--api_key", type=str, default=None, help="Optional NCBI API key")
    parser.add_argument("--tool", type=str, default="clinical_kb_builder")
    parser.add_argument("--sort", type=str, default="relevance", choices=["relevance", "pub date"])

    # selection size
    parser.add_argument("--max_per_query", type=int, default=50)
    parser.add_argument("--max_total_articles", type=int, default=200)

    # chroma
    parser.add_argument("--chroma_host", type=str, default="localhost")
    parser.add_argument("--chroma_port", type=int, default=8000)
    parser.add_argument("--collection_name", type=str, default="pmc_kb")

    # embedding
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch_size", type=int, default=32)

    # chunking
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--min_chunk_chars", type=int, default=250)

    # output
    parser.add_argument("--manifest_out", type=str, default="pmc_kb_manifest.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pmc_kb(args)


if __name__ == "__main__":
    main()
