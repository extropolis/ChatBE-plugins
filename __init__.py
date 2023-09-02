from .base import BaseTool
from .local_search.local_search import LocalSearchTool
from .news_web_search.news_search import NewsSearchTool
from .news_web_search.web_search import WebSearchTool
from .memory.memory import MemoryTool
from .file_process.file_process import FileProcessTool
from .image_creation.image_creation import ImageCreation

__all__ = [
    "BaseTool",
    "LocalSearchTool",
    "NewsSearchTool",
    "WebSearchTool",
    "MemoryTool",
    "FileProcessTool",
    "ImageCreation",
]

name_map = {
    LocalSearchTool.name: "LocalSearchTool",
    NewsSearchTool.name: "NewsSearchTool",
    WebSearchTool.name: "WebSearchTool",
    MemoryTool.name: "MemoryTool",
    FileProcessTool.name: "FileProcessTool",
    ImageCreation.name: "ImageCreation",
}
