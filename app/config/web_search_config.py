from __future__ import annotations


# app/config/web_search_config.py
"""
Web Search configuration for CookHero.
Uses Tavily API for web search.
"""

from typing import Optional

from pydantic import BaseModel


class WebSearchConfig(BaseModel):
    """
    Configuration for web search functionality using Tavily.
    """

    enabled: bool = True
    api_key: Optional[str] = None  # Loaded from .env (WEB_SEARCH_API_KEY)
    max_results: int = 5
