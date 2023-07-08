from .base import BaseTool
from .local_search.local_search import LocalSearchTool
from .news_web_search.news_search import NewsSearchTool
from .news_web_search.web_search import WebSearchTool
from .memory.memory import MemoryTool

__all__ = [
    "BaseTool",
    "LocalSearchTool",
    "NewsSearchTool",
    "WebSearchTool",
    "MemoryTool",
]

name_map = {
    LocalSearchTool.name: "LocalSearchTool",
    NewsSearchTool.name: "NewsSearchTool",
    WebSearchTool.name: "WebSearchTool",
    MemoryTool.name: "MemoryTool",
}
