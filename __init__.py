from tools.base import BaseTool
from tools.local_search import LocalSearchTool
from tools.news_web_search.news_search import NewsSearchTool
from tools.news_web_search.web_search import WebSearchTool
from tools.memory import MemoryTool

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
