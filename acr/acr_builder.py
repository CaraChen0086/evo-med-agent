"""
ACR Builder Module.

Orchestrates ACR search and parsing to build evidence items.
"""

from typing import List, Dict
from .acr_search import ACRSearch
from .acr_parser import ACRParser


class ACRBuilder:
    """Builds ACR evidence from queries."""

    def __init__(self, search_timeout: int = 30, parse_timeout: int = 30):
        self.search = ACRSearch(timeout=search_timeout)
        self.parser = ACRParser(timeout=parse_timeout)

    def build_evidence_from_query(self, query: str, top_k_topics: int = 3, max_scenarios_per_topic: int = 3) -> List[Dict]:
        """
        Build ACR evidence from a query.

        Args:
            query: Search keyword
            top_k_topics: Number of top topics to process
            max_scenarios_per_topic: Max scenarios/docs per topic

        Returns:
            List of ACR items with parsed data.
        """
        # Search topics
        topics = self.search.search_topics(query)
        if not topics:
            print(f"[ACR Builder] No topics found for query: {query}")
            return []

        # Limit to top-k
        topics = topics[:top_k_topics]
        acr_items = []

        for topic in topics:
            print(f"[ACR Builder] Processing topic: {topic['topic']}")

            # Process docs (limit per topic)
            docs_to_process = topic['docs'][:max_scenarios_per_topic]

            for doc in docs_to_process:
                url = doc['url']
                if not url.startswith('http'):
                    continue  # Skip invalid URLs

                parsed = self.parser.parse_page(url)
                if 'error' not in parsed and parsed.get('recommendations'):
                    # Merge topic info
                    item = {
                        **parsed,
                        "panel": topic['panel'],
                        "topic_name": topic['topic'],
                        "doc_label": doc['label']
                    }
                    acr_items.append(item)
                    print(f"[ACR Builder] Parsed {len(parsed['recommendations'])} recommendations from {url}")

        return acr_items

    def acr_item_to_chunks(self, item: Dict) -> List[str]:
        """
        Convert ACR item to retrieval-friendly chunk texts.

        Args:
            item: Parsed ACR item

        Returns:
            List of chunk strings.
        """
        chunks = []
        base_info = f"[SourceType: ACR Rating Table]\n[Topic: {item.get('topic_name', 'Unknown')}]\n[Panel: {item.get('panel', 'Unknown')}]"

        if item.get('scenario_text'):
            base_info += f"\n[Scenario: {item['scenario_text']}]"

        if item.get('variant_text'):
            base_info += f"\n[Variant: {item['variant_text']}]"

        for rec in item.get('recommendations', []):
            chunk = f"{base_info}\nProcedure: {rec['procedure']}\nAppropriateness: {rec['appropriateness']}"
            chunks.append(chunk)

        return chunks