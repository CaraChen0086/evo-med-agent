"""
ACR Parser Module.

Parses individual ACR pages to extract structured recommendations.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re


class ACRParser:
    """Parses ACR scenario/topic pages for structured data."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ACR Parser Tool/1.0"
        })

    def parse_page(self, url: str) -> Dict:
        """
        Parse an ACR page URL for structured data.

        Args:
            url: Full URL to ACR page

        Returns:
            Dict with topic, scenario_text, variant_text, recommendations, etc.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ACR Parser] Error fetching page {url}: {e}")
            return {"url": url, "error": str(e)}

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract topic
        topic = self._extract_topic(soup)

        # Extract scenario text (if available)
        scenario_text = self._extract_scenario_text(soup)

        # Extract variant text (if available)
        variant_text = self._extract_variant_text(soup)

        # Extract recommendations from rating table
        recommendations = self._extract_recommendations(soup)

        return {
            "url": url,
            "topic": topic,
            "scenario_text": scenario_text,
            "variant_text": variant_text,
            "recommendations": recommendations
        }

    def _extract_topic(self, soup: BeautifulSoup) -> str:
        """Extract topic name."""
        title = soup.find('h1') or soup.find('title')
        return title.get_text(strip=True) if title else "Unknown Topic"

    def _extract_scenario_text(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract scenario description (defensive)."""
        scenario_div = soup.find('div', class_='scenario-text')
        return scenario_div.get_text(strip=True) if scenario_div else None

    def _extract_variant_text(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract variant description (defensive)."""
        variant_div = soup.find('div', class_='variant-text')
        return variant_div.get_text(strip=True) if variant_div else None

    def _extract_recommendations(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract procedure recommendations from rating table."""
        recommendations = []

        # Look for rating table (common patterns)
        table = soup.find('table', class_=re.compile(r'rating|appropriateness'))
        if not table:
            # Fallback: look for any table with procedure/appropriateness headers
            tables = soup.find_all('table')
            for t in tables:
                headers = [th.get_text(strip=True).lower() for th in t.find_all('th')]
                if 'procedure' in headers and any('appropriate' in h for h in headers):
                    table = t
                    break

        if not table:
            print("[ACR Parser] No rating table found")
            return recommendations

        # Parse table rows
        rows = table.find_all('tr')[1:]  # Skip header
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                procedure = cells[0].get_text(strip=True)
                appropriateness = cells[1].get_text(strip=True)
                if procedure and appropriateness:
                    recommendations.append({
                        "procedure": procedure,
                        "appropriateness": appropriateness
                    })

        return recommendations