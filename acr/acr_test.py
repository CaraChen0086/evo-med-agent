import json
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://acsearch.acr.org"
LIST_URL = f"{BASE_URL}/list"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DOC_LABELS = {
    "Narrative & Rating Table",
    "Evidence Table",
    "Lit Search",
    "Appendix",
    "Patient Summary",
}

SKIP_TEXTS = {
    "",
    "SEARCH",
    "CLEAR",
    "En Español",
    "Diagnostic",
    "Interventional",
    "Panels",
    "CMS Priority Clinical Areas",
}

PANEL_NAMES = {
    "Breast",
    "Cardiac",
    "Gastrointestinal",
    "General and Vascular",
    "Musculoskeletal",
    "Neurological",
    "Pediatric",
    "Radiation Oncology",
    "Thoracic",
    "Urologic",
    "Women's Imaging",
}

def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def is_probable_topic(text: str) -> bool:
    text = clean_text(text)

    if text in SKIP_TEXTS:
        return False

    if text in DOC_LABELS:
        return False

    if text in PANEL_NAMES:
        return False

    if len(text) < 3 or len(text) > 140:
        return False

    # 避免把一些控制文字当 topic
    bad_keywords = [
        "Total Records",
        "Filtered",
        "Search",
        "Clear",
        "Panel",
        "Scenario",
        "Sex",
        "Age",
        "Body area",
    ]
    if any(bad.lower() in text.lower() for bad in bad_keywords):
        return False

    return True


def parse_acr_topics(html: str):
    soup = BeautifulSoup(html, "html.parser")

    results = []
    current_panel = None
    current_topic = None
    current_docs = []

    def flush_current():
        nonlocal current_topic, current_docs
        if current_panel and current_topic:
            results.append(
                {
                    "panel": current_panel,
                    "topic": current_topic,
                    "docs": current_docs[:],
                }
            )
        current_topic = None
        current_docs = []

    # 取页面里所有有文本或链接意义的元素
    elements = soup.find_all(["h1", "h2", "h3", "a", "span", "div", "p"])

    for el in elements:
        if not isinstance(el, Tag):
            continue

        text = clean_text(el.get_text(" ", strip=True))
        if not text:
            continue

        # 1) Panel
        if el.name in {"h1", "h2", "h3"} and text in PANEL_NAMES:
            flush_current()
            current_panel = text
            continue

        # 2) 文档链接
        if el.name == "a":
            link_text = text
            if link_text in DOC_LABELS:
                href = el.get("href")
                if current_topic and href:
                    current_docs.append(
                        {
                            "label": link_text,
                            "url": urljoin(BASE_URL, href),
                        }
                    )
                continue

        # 3) topic 候选
        # 尽量只在已进入某个 panel 后开始识别 topic
        if current_panel and is_probable_topic(text):
            # 避免把 "topic + doc labels" 混进去
            if any(label in text for label in DOC_LABELS):
                continue

            # 新 topic 出现，先保存旧的
            if current_topic and text != current_topic:
                flush_current()

            if current_topic is None:
                current_topic = text

    flush_current()

    # 去重：有些页面结构会重复抓到 topic
    deduped = []
    seen = set()
    for item in results:
        key = (item["panel"], item["topic"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def search_topics(topics, keyword: str):
    kw = clean_text(keyword).lower()
    tokens = kw.split()

    matches = []
    for item in topics:
        combined = f"{item['panel']} {item['topic']}".lower()

        # 宽松匹配：所有 token 都出现
        if all(token in combined for token in tokens):
            matches.append(item)

    return matches


def main():
    keyword = "breast cancer"

    print(f"Fetching ACR list page: {LIST_URL}")
    html = fetch_html(LIST_URL)

    print("Parsing topics...")
    topics = parse_acr_topics(html)

    print("Searching keyword...")
    matches = search_topics(topics, keyword)

    print(f"Total parsed topics: {len(topics)}")
    print(f"Keyword: {keyword!r}")
    print(f"Matched topics: {len(matches)}")
    print("-" * 80)

    for i, item in enumerate(matches, 1):
        print(f"[{i}] Panel: {item['panel']}")
        print(f"    Topic: {item['topic']}")
        print(f"    Document count: {len(item['docs'])}")
        for doc in item["docs"]:
            print(f"      - {doc['label']}: {doc['url']}")
        print()

    output_path = "acr_topics_breast_cancer.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()