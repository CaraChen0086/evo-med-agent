"""
ACR Search Module.

Searches the ACR appropriateness criteria list page for topics.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import sys


class ACRSearch:
    """Handles searching ACR topics from the public list page."""

    BASE_URL = "https://acsearch.acr.org/list"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ACR Search Tool/1.0"
        })

    def search_topics(self, keyword: str) -> List[Dict]:
        """
        Search for ACR topics matching the keyword.

        Args:
            keyword: Search term (e.g., "breast cancer")

        Returns:
            List of topic dicts with 'panel', 'topic', and 'docs' keys.
        """
        params = {"keyword": keyword}
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ACR Search] Error fetching topics: {e}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        topics = []

        # Parse topic entries (defensive parsing)
        for topic_div in soup.find_all('div', class_='topic-entry'):
            try:
                panel = topic_div.find('h3', class_='panel-title')
                panel_text = panel.get_text(strip=True) if panel else "Unknown Panel"

                topic_link = topic_div.find('a', class_='topic-link')
                if not topic_link:
                    continue
                topic_text = topic_link.get_text(strip=True)
                topic_url = topic_link.get('href')

                # Extract docs (narrative, evidence table, etc.)
                docs = []
                doc_links = topic_div.find_all('a', href=True)
                for link in doc_links:
                    if link != topic_link:  # Skip main topic link
                        docs.append({
                            "label": link.get_text(strip=True),
                            "url": link.get('href')
                        })

                topics.append({
                    "panel": panel_text,
                    "topic": topic_text,
                    "docs": docs
                })
            except Exception as e:
                print(f"[ACR Search] Error parsing topic: {e}")
                continue

        return topics


def cli_search(keyword: str):
    """CLI entrypoint for manual testing."""
    search = ACRSearch()
    topics = search.search_topics(keyword)
    print(f"Found {len(topics)} topics for '{keyword}':")
    for i, topic in enumerate(topics[:5], 1):  # Limit output
        print(f"{i}. {topic['topic']} (Panel: {topic['panel']})")
        for doc in topic['docs']:
            print(f"   - {doc['label']}: {doc['url']}")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python acr_search.py <keyword>")
        sys.exit(1)
    cli_search(sys.argv[1])