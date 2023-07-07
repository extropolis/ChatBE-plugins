import re
import unicodedata
from newspaper import Article, Config

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