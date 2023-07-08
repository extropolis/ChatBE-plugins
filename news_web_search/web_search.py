import os
from typing import Callable
from serpapi import GoogleSearch
from ..base import BaseTool

def web_search(query, location = None, time_range = 'd', n_results=5):
    """
        Direct call on serpapi to retrieve raw document
        query: search term
        content_type: text, image, video, etc...
        location: news location to facilitate search quality
        Returns:
            a list of n_results news, each news is represented by the following JSON object: 
            {
                "title": <string>,
                "content": <string> basic snippet for now,
                "source": <string> link to the article
            }
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.environ["SERPAPI_KEY"],
        "tbs": f"qdr:{time_range},sbd:1"
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
    except:
        return []
    
    search_results = []
    if "answer_box" in results:
        r = results["answer_box"]
        if "answer" in r:
            content = r["answer"]
        elif "snippet" in r:
            content = r["snippet"]
        search_results.append({
            "title": "google answer box",
            "content": content,
            "source": "www.google.com",
        })
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
        search_results.append({
            "title": "google knowledge graph",
            "content": results["knowledge_graph"]["description"],
            "source": "www.google.com",
        })
    if "organic_results" in results and len(results["organic_results"]) > 0:
        for r in results["organic_results"]:
            if "link" not in r or ("snippet" not in r and "title" not in r):
                continue
            curr_result = {
                "title": r["title"] if "title" in r else r["snippet"],
                "content": r["snippet"] if "snippet" in r else r["title"],
                "source": r["link"]
            }
            search_results.append(curr_result)
            if len(search_results) >= n_results:
                break
    return search_results

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "tool for general web search"
    user_description: str = "You can enable this to perform general web search."
    def __init__(self, func: Callable=None, **kwargs):
        '''The web search must be fast, and could be simple general information such as title or snippet'''
        self.n_results = kwargs.get("n_search_results", 5)
        super().__init__(None)
        self.args["properties"]["req_info"]["description"] = "User's current location. You can have it as a place holder"
        self.args["properties"]["query"]["description"] = "The string used to search. Make it as concise as possible."
        self.args["required"] = ["query"]
    
    def _run(self, query: str, req_info: dict=None):
        search_results = web_search(query, self.n_results)
        if len(search_results) == 0:
            # Hopefully it will never reach this state
            return []
        return search_results