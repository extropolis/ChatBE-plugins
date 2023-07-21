import re, os, threading
import unicodedata
from typing import Callable
from newspaper import Article, Config
from serpapi import GoogleSearch
from ..base import BaseTool
try:
    from .retrieval import STArticleRetriver
    use_retriver = True
except:
    use_retriver = False

class Newspaper3KParser():
    def __init__(self, url):
        config = Config()
        config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        config.request_timeout = 10
        self.article = Article(url, config=config)

    def parse(self):
        self.article.download()
        self.article.parse()
        self.body = self.get_body()
        self.title = self.article.title
        self.word_count = self.get_word_count()

    def post_proc(self, text):
        """remove white space, remove empty string, unescape html/unicode"""
        text = re.sub('\s\s+', ' ', text.strip())
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r"(http|www)\S+", "", text)
        text = text.strip()
        return text

    def get_body(self):
        body = self.article.text.split("\n\n")
        for i in range(len(body)):
            body[i] = self.post_proc(body[i])
        return body

    def get_title(self):
        return self.article.title

    def get_word_count(self):
        count = 0
        for b in self.body:
            count += len(b.split())
        return count

def news_search_multi(query, content_type=None, location = None, time_range = 'd', results=[]):
    if content_type is None:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.environ["SERPAPI_KEY"],
            "tbs": f"qdr:{time_range},sbd:1"
        }
    else:
        params = {
            "engine": "google",
            "q": query,
            "tbm": content_type,
            "api_key": os.environ["SERPAPI_KEY"],
            "tbs": f"qdr:{time_range},sbd:1"
        }
    try:
        search = GoogleSearch(params)
        results.append(search.get_dict())
    except:
        return
    
def news_search(query, location = None, time_range = 'd', n_results=5):
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
    results = []
    threads = []

    for content_type in [None, "nws"]:
        t = threading.Thread(target=lambda:news_search_multi(query, content_type, results=results, location=location, time_range=time_range))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    res = results[0]
    res.update(results[1])
    results = res

    search_results = []

    # give news search a higher priority
    if "news_results" in results and len(results["news_results"]) > 0:
        for news in results["news_results"]:
            if "link" not in news or ("snippet" not in news and "title" not in news):
                continue
            curr_result = {
                "title": news["title"] if "title" in news else news["snippet"],
                "content": news["snippet"] if "snippet" in news else news["title"],
                "source": news["link"]
            }
            search_results.append(curr_result)
            if len(search_results) >= n_results:
                break

    if len(search_results) >= n_results:
        return search_results
    
    if "organic_results" in results and len(results["organic_results"]) > 0:
        for news in results["organic_results"]:
            if "link" not in news or ("snippet" not in news and "title" not in news):
                continue
            curr_result = {
                "title": news["title"] if "title" in news else news["snippet"],
                "content": news["snippet"] if "snippet" in news else news["title"],
                "source": news["link"]
            }
            search_results.append(curr_result)
            if len(search_results) >= n_results:
                break
    
    return search_results

class NewsSearchTool(BaseTool):
    name: str = "news_search"
    # description: str = "Search latest news given an explicit query. If the user is asking news about you yourself or the user (e.g. What's new with you?, What's new for you?), you don't need to use it. If the user is asking about history or general common knowledge (e.g. general introduction or history), even if could be updated after the year 2021, you should not use it."
    description: str = "Search for the latest news given an explicit query. Not intended for questions about the user, the AI assistant or you yourself (e.g. What's new with you?, What's new for you?), or for general knowledge questions."
    user_description: str = "You can enable this to search for latest news."
    def __init__(self, func: Callable=None, **kwargs):
        '''
            If there is a retriever model provided, the news search tool will try to get the content of the news rather than a simple snippet or title.
        '''
        path = kwargs.get("news_model_path", os.path.join(os.path.abspath(os.path.curdir), "tools/news_web_search/models/tart_ds_march_8/"))
        self.n_results = kwargs.get("n_search_results", 5)
        if not os.path.exists(path) or not use_retriver:
            self.retriever = None
        else:
            self.retriever = STArticleRetriver(path)
        super().__init__(None)
        self.args["properties"]["req_info"]["description"] = "User's current location. You can have it as a place holder"
        self.args["properties"]["query"]["description"] = "The string used to search. Make it as concise as possible. If you don't need any specific query, put \"What's new\" for query. Also try to integrate one and only one user interest if appropriate. e.g. {\"user\": \"What's new?\"}, user_interest: [\"sports\", \"tech\"], query: What's new in tech?"
        self.args["required"] = ["query"]
    
    def parsing_multi(self, search_results, i):
        try:
            news_parser = Newspaper3KParser(search_results[i]["source"])
            news_parser.parse()
            retrieved = self.retriever.retrive_most_relevant(title=news_parser.title, text=news_parser.body)
            search_results[i]["content"] += "\n" + "\n".join(retrieved).strip()
        except Exception as e:
            print(e)
    
    def _run(self, query: str, req_info: dict=None):
        if query == "":
            query = "What's new?"
        search_results = news_search(query, self.n_results)
        if len(search_results) == 0:
            # Hopefully it will never reach this state
            return search_results
        if self.retriever is not None:
            # We have a retriever, then we can replace the news content with some more relevant stuff
            threads = []
            for i in range(len(search_results)):
                # t = threading.Thread(target=lambda:self.parsing_multi(search_results, i))
                # t.start()
                # threads.append(t)
                self.parsing_multi(search_results, i)
            for t in threads:
                t.join()
        return search_results