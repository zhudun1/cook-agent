"""
Web Search Tool for CookHero.

Provides two core methods:
1. decide_search() - Determines if web search is needed and generates search parameters
2. execute_search() - Executes the actual web search using Tavily API

Uses Tavily official Python client for reliable web search.
Uses LLM tool calling for structured output.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.config import settings, LLMType
from app.llm import LLMProvider, get_usage_callbacks, llm_context

logger = logging.getLogger(__name__)

THRESHOLD_CONFIDENCE = 6  # Confidence threshold to decide if search is needed


class SearchDecisionInput(BaseModel):
    """Input schema for search decision."""

    confidence: int = Field(
        description="Confidence score from 0-10, higher means more likely to need web search. 0-5: No web search needed, 6-10: Web search recommended",
        ge=0,
        le=10,
    )
    search_query: str = Field(
        description="Optimized search keywords (concise and precise) suitable for web search. The number of words should be kept minimal and focused.",
    )
    reason: str = Field(
        description="Brief explanation of why web search is or isn't needed"
    )


@dataclass
class WebSearchParams:
    """Parameters for executing a web search."""

    query: str
    max_results: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "max_results": self.max_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSearchParams":
        return cls(
            query=data.get("query", ""),
            max_results=data.get("max_results", 5),
        )


@dataclass
class WebSearchDecision:
    """Result of web search decision."""

    confidence: int  # 0-10, higher means more likely to need web search
    search_params: Optional[WebSearchParams] = None
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_search(self) -> bool:
        """Check if confidence meets threshold for searching."""
        return self.confidence >= THRESHOLD_CONFIDENCE


@dataclass
class WebSearchResult:
    """A single web search result."""

    title: str
    snippet: str  # Summary or key information
    source: str  # Site name or identifier
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "url": self.url,
        }


# Prompt template for web search decision
WEB_SEARCH_DECISION_PROMPT_TEMPLATE = """
你是 CookHero 的「Web 搜索决策模块」，专门判断当前用户问题是否需要进行互联网搜索来补充回答。

【决策原则】

需要 Web 搜索（confidence 应该较高，6-10）的情况：
1. **时效性信息**
   - 询问最近的美食新闻、餐厅推荐、食材价格趋势
   - 涉及季节性食材的当前市场情况
2. **本地知识库可能不足的内容**
   - 非常规或小众菜系的详细做法
   - 特定品牌产品的使用方法
   - 需要最新研究支持的营养健康信息
3. **用户明确要求搜索网络**
   - 用户提到"搜索一下"、"网上查查"等
4. **需要对比多来源信息**
   - 用户要求比较不同做法或观点

不需要 Web 搜索（confidence 应该较低，0-5）的情况：
1. **常规烹饪问题**
   - 经典菜谱、基础烹饪技巧
   - 常见食材处理方法
2. **对话延续**
   - 闲聊、确认、追问细节
   - 基于上下文的后续问题
3. **本地知识库足以回答**
   - 标准家常菜做法
   - 基础烹饪原理

【本地知识库已有的信息】
{document_summary}

{history}
"""

WEB_SEARCH_DECISION_PROMPT = ChatPromptTemplate.from_template(
    WEB_SEARCH_DECISION_PROMPT_TEMPLATE
)


class WebSearchTool:
    """
    Web Search Tool providing decision and execution methods.

    Uses Tavily official Python client for web search.
    """

    MODULE_NAME = "web_search"

    def __init__(
        self,
        llm_type: LLMType | str = LLMType.FAST,
        api_key: Optional[str] = None,
        max_results: Optional[int] = None,
        provider: LLMProvider | None = None,
    ):
        """
        Initialize the Web Search Tool.

        Args:
            llm_type: Which LLM tier to use (fast/normal)
            api_key: Tavily API key
            max_results: Maximum search results to return
        """
        # Load from settings with overrides
        web_search_config = settings.web_search

        self.api_key = (
            api_key or web_search_config.api_key or os.getenv("WEB_SEARCH_API_KEY", "")
        )
        self.max_results = max_results or web_search_config.max_results
        self.enabled = web_search_config.enabled

        # Initialize Tavily client (lazy initialization)
        self._tavily_client: Optional[TavilyClient] = None

        # Initialize LLM for decision making
        self._llm_type = llm_type
        self._provider = provider or LLMProvider(settings.llm)

        # Create decision tool
        self._decision_tool = self._create_decision_tool()

        # Create LLM with callbacks and bind tools
        # Use tracked invoker for usage statistics
        self._callbacks = get_usage_callbacks()
        base_llm = self._provider.create_llm(llm_type, temperature=0.3)
        self._llm = base_llm.bind_tools(
            [self._decision_tool], tool_choice="make_search_decision"
        )

    def _create_decision_tool(self):
        """Create the tool for search decision using @tool decorator."""

        @tool(args_schema=SearchDecisionInput)
        def make_search_decision(
            confidence: int,
            search_query: str,
            reason: str,
        ) -> dict:
            """
            Make a decision on whether web search is needed.

            Args:
                confidence: Confidence score from 0-10, higher means more likely to need web search
                search_query: Optimized search keywords (concise and precise)
                reason: Brief explanation of why web search is or isn't needed

            Returns:
                Dictionary containing the decision details
            """
            return {
                "confidence": confidence,
                "search_query": search_query,
                "reason": reason,
            }

        return make_search_decision

    @property
    def tavily_client(self) -> Optional[TavilyClient]:
        """Lazy initialization of Tavily client."""
        if self._tavily_client is None and self.api_key:
            try:
                self._tavily_client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
        return self._tavily_client

    async def decide_search(
        self,
        query: str,
        document_summary: Dict[str, List[str]],
        history_text: str = "",
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> WebSearchDecision:
        """
        Decide whether web search is needed and generate search parameters.

        Args:
            query: Current user query
            document_summary: Summary of documents in local knowledge base
            history_text: Formatted conversation history
            user_id: User ID for tracking (optional)
            conversation_id: Conversation ID for tracking (optional)

        Returns:
            WebSearchDecision with confidence score and search parameters
        """
        try:
            # Format document summary
            document_summary_str = ""
            if document_summary:
                dishes = document_summary.get("dish_name", [])
                document_summary_str = "已知菜品名称: " + ", ".join(dishes) + "\n"

            # Format prompt
            prompt = WEB_SEARCH_DECISION_PROMPT.format_prompt(
                history=history_text,
                document_summary=document_summary_str,
            )
            # Use llm_context for usage tracking
            with llm_context(self.MODULE_NAME, user_id, conversation_id):
                response = await self._llm.with_config(
                    callbacks=self._callbacks
                ).ainvoke(prompt.messages)

            # Parse tool calls
            if not response.tool_calls:
                logger.warning("No tool call in response, returning low confidence")
                return WebSearchDecision(
                    confidence=0,
                    search_params=None,
                    reason="LLM未调用决策工具",
                    raw={},
                )

            # Get first tool call result
            tool_call = response.tool_calls[0]
            args = tool_call["args"]

            # Extract and validate confidence
            confidence = int(args.get("confidence", 0))
            confidence = max(0, min(10, confidence))  # Clamp to 0-10

            # Extract search query
            search_query = args.get("search_query", query)
            reason = args.get("reason", "")

            # Create search params
            search_params = WebSearchParams(
                query=search_query,
                max_results=self.max_results,
            )

            return WebSearchDecision(
                confidence=confidence,
                search_params=search_params,
                reason=reason,
                raw=args,
            )

        except Exception as e:
            logger.error(f"Web search decision failed: {e}", exc_info=True)
            # Return low confidence on error
            return WebSearchDecision(
                confidence=0,
                search_params=None,
                reason=f"Decision failed: {str(e)[:50]}",
                raw={},
            )

    async def execute_search(
        self,
        search_params: WebSearchParams,
    ) -> List[WebSearchResult]:
        """
        Execute web search using Tavily API.

        Args:
            search_params: Parameters for the search

        Returns:
            List of WebSearchResult objects
        """
        if not self.tavily_client:
            logger.warning("Tavily client not initialized, returning empty results")
            return []

        try:
            # Use Tavily's search method
            response = self.tavily_client.search(
                query=search_params.query,
                topic="general",
                search_depth="basic",
                max_results=search_params.max_results,
                include_answer=False,
                include_images=False,
                include_raw_content=True,
            )

            results = []
            for item in response.get("results", [])[: search_params.max_results]:
                results.append(
                    WebSearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        source=self._extract_domain(item.get("url", "")),
                        url=item.get("url"),
                    )
                )

            logger.info(
                f"Tavily search completed: query='{search_params.query}', results={len(results)}"
            )
            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}", exc_info=True)
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL for source identification."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "web"

    def format_results_for_context(
        self,
        results: List[WebSearchResult],
        max_length: int = 2000,
    ) -> str:
        """
        Format search results for inclusion in LLM context.

        Args:
            results: List of search results
            max_length: Maximum total character length

        Returns:
            Formatted string for context injection
        """
        if not results:
            return ""

        lines = []
        current_length = 0

        for i, result in enumerate(results, 1):
            entry = f"[{i}] {result.title}\n来源: {result.source}\n{result.snippet}"
            if result.url:
                entry += f"\n链接: {result.url}"
            entry += "\n"

            if current_length + len(entry) > max_length:
                break

            lines.append(entry)
            current_length += len(entry)

        return "\n".join(lines)


# Singleton instance
web_search_tool = WebSearchTool()
