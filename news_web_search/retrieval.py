import os
import math
import torch
import torch.nn.functional as F
import logging
import pandas as pd
from torch.utils.data import DataLoader
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses, util
# from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
# from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
from tools.news_web_search.news_parser import Newspaper3KParser
PERF_TEST = False
LOG_DIR = "logs/"
USE_THD = 0.6
TOP_PCT = 0.1
BOT_THD = 0.1

import os

if not os.path.exists("logs/"):
    os.makedirs("logs/")


def training_set_builder_from_tart(article_link_file: str, save_path: str):
    assert os.path.exists(article_link_file)
    tart = TartArticleRetriver()
    with open(article_link_file, 'r') as r:
        links = r.readlines()
    title, paragraph, label = [], [], []
    for link in links:
        parsed = Newspaper3KParser(link.strip())
        parsed.parse()
        scores = tart.retrive_most_relevant(
            title=parsed.title, text=parsed.body, limit=-1)
        for score in scores:
            title.append(parsed.title)
            paragraph.append(score[1])
            label.append(score[2])
    df = pd.DataFrame(
        data={'title': title, 'paragraph': paragraph, 'label': label})
    df.to_csv(save_path)
    return

def retriver_factory(type: str, **kwargs):
    for cls in BaseRetriver.__subclasses__():
        if cls.is_registrar_for(type):
            return cls(**kwargs)
    return BaseRetriver()

class BaseRetriver():
    def __init__(self, logging_name, logging_level=10) -> None:
        self.set_logger(logging_name, logging_level)
        self.use_cuda = self.use_cuda = torch.cuda.is_available()

    def set_logger(self, logging_name, logging_level):
        FORMAT = '\n#####\n%(asctime)s %(message)s\n#####\n'
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        logging.basicConfig(
            filename=os.path.join(LOG_DIR, logging_name),
            filemode='a',
            format=FORMAT
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def retrive_most_relevant(self, title, text, limit, thres):
        raise NotImplementedError()


class TartArticleRetriver(BaseRetriver):
    def __init__(self, instr: str = "Retrieve a passage that is the most relevant to the title", logging_level=10):
        '''
            Input:
                instr: the rule to retrive relavent paragraphs
        '''
        logging_name = "Tart_retrival.log"
        super().__init__(logging_name=logging_name, logging_level=logging_level)

        self.model = EncT5ForSequenceClassification.from_pretrained(
            "facebook/tart-full-flan-t5-xl",use_auth_token=True)
        # if self.use_cuda:
        #     self.model.cuda()
        self.model.eval()
        self.tokenizer = EncT5Tokenizer.from_pretrained(
            "facebook/tart-full-flan-t5-xl",use_auth_token=True)
        self.instr = instr

    @classmethod
    def is_registrar_for(cls, type):
        return type == "TART"
    

    def format_inputs(self, text, query, thres):
        useful_text = []
        for t in text:
            if len(t.split()) > thres:
                useful_text.append(t)

        input_seq = []
        for _ in range(len(useful_text)):
            input_seq.append("{0} [SEP] {1}".format(self.instr, query))
        return input_seq, useful_text


    def tokenize(self, input_seq, useful_text):
        # need to tweak the padding, truncation parameters
        features = self.tokenizer(
            input_seq, useful_text, padding=True, truncation=False, return_tensors="pt")
        # if self.use_cuda:
        #     features = features.to("cuda")
        return features


    def inference(self, features, useful_text):
        with torch.no_grad():
            scores = self.model(**features)
            scores = scores.logits.cpu()
            normalized_scores = [float(score[1])
                                 for score in F.softmax(scores, dim=1)]

        score_dict = list(zip(list(range(len(useful_text))),
                          useful_text, normalized_scores))
        sorted_scores = sorted(
            score_dict, key=lambda v: v[2], reverse=True)
        return sorted_scores

    def retrive_most_relevant(self, title: str, text: List[str], limit=3, thres=10):
        '''
            Given title and a list of paragraphs, find the most relevant paragraphs

            Input:
                title: the title of the article
                text: the main body part of the article
                limit: the number of paragraphs to return, default 3, meaning that the top 3 most relevant paragraphs will be returned.
                thres: the least number of words for paragraphs to be considered. If a paragraph has less than thres words, it will not be considered. Default: 10
                logging: whether or not log to the terminal
        '''
        assert thres >= 0

        input_seq, useful_text = self.format_inputs(text, title, thres)
        features = self.tokenize(input_seq, useful_text)
        sorted_scores = self.inference(features, useful_text)

        self.logger.debug(
            ("\n".join([f"prob:{s[2]}\t|sequence:{s[0]}\n{s[1]}" for s in sorted_scores])))
        if limit <= 0:
            return sorted_scores
        else:
            result = sorted_scores[:limit]
        result = sorted(result, key=lambda x: x[0])
        return [x[1] for x in result]


class STArticleRetriver(BaseRetriver):
    def __init__(self, load_pretrained: str = None, logging_level=10):
        logging_name = "st_retriver.log"
        super().__init__(logging_name=logging_name, logging_level=logging_level)
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.base_model = "all-mpnet-base-v2"
        if load_pretrained:
            self.model = SentenceTransformer(
                load_pretrained, device=self.device)
            self.model_name = load_pretrained
        else:
            self.model = SentenceTransformer(
                self.base_model, device=self.device)
            self.model_name = self.base_model

    @classmethod
    def is_registrar_for(cls, type):
        return type == "ST"
    
    def load_dataset(self, dataset_path: str, batch_size=16):
        df = pd.read_csv(dataset_path)
        train_examples = []
        for _, row in df.iterrows():
            train_examples.append(InputExample(
                texts=[row['title'], row['paragraph']], label=row['label']))
        dataset = DataLoader(train_examples, shuffle=True,
                             batch_size=batch_size)
        return dataset

    def train(self, dataset_path: str, epoch=10, save_path=None):
        try:
            ds = self.load_dataset(dataset_path=dataset_path)
            train_loss = losses.ContrastiveLoss(model=self.model)
            self.model.fit(
                train_objectives=[(ds, train_loss)],
                epochs=epoch,
                warmup_steps=100,
                output_path=save_path
            )
        except Exception as e:
            self.logger.error(f"Error: Training failed! Using base model. {e}")
            self.model = SentenceTransformer(
                self.base_model, device=self.device)
            self.model_name = self.base_model

    def retrive_most_relevant(self, title: str, text: List[str], limit=3, thres=10, top_pct=TOP_PCT):
        useful_text, result = [], []
        for t in text:
            if len(t.split()) > thres:
                useful_text.append(t)
        useful_text.append(title)
        top_k = math.ceil((len(useful_text) - 1)*top_pct)
        embeddings = self.model.encode(useful_text)
        cosine_scores = util.cos_sim(embeddings[:-1], embeddings[-1])
        sequence = list(range(len(cosine_scores)))
        _sorted_score = sorted(list(zip(cosine_scores, sequence)), key=lambda x:x[0], reverse=True)
        if len(_sorted_score) == 0:
            return []
        best_p = _sorted_score[0][1]
        
        p_cosine_scores = util.cos_sim(embeddings[:-1], embeddings[best_p])
        self.max_para = len([x for x in p_cosine_scores if x >BOT_THD])
        _sorted_score = sorted(list(zip(p_cosine_scores, sequence)), key=lambda x:x[0], reverse=True) #(score, sequence)
        self.logger.debug(f"TS model[{self.model_name}] similarity: \n" + "\n".join(
            [f"p_prob:{cs}\n{t}\n" for cs, t in zip(p_cosine_scores, useful_text)]))
        ret = sorted(_sorted_score[:top_k], key=lambda x:x[1])
        self.last_article = [(*x, useful_text[x[1]]) for x in _sorted_score] #(score, sequence, text)
        self.read = top_k
        self.top_k = top_k 
        return [useful_text[x[1]] for x in ret]

if __name__ == "__main__":
    import time
    # training_set_builder_from_tart('./data/training_links', './data/retriver_data.csv')
    # # start = time.time()
    # link = "https://www.ctvnews.ca/politics/what-is-bill-c-18-and-how-do-i-know-if-google-is-blocking-my-news-content-1.6286816"
    # # # link = "https://www.nytimes.com/2023/03/18/opinion/woke-definition.html"
    # # # nyt has some anti bot mechenism, cannot pass through.
    # parsed = Newspaper3KParser(link)
    # parsed.parse()
    # end = time.time()
    # print(f"word count: {parsed.word_count}")

    # start = time.time()
    # retriver = retriver_factory('TART')
    # r = retriver.retrive_most_relevant(
    # title=parsed.title, text=parsed.body)
    # end = time.time()
    # print(f"TART retrive time {end-start:.3f}s")
    # print(r)
    # retriver = STArticleRetriver()
    # print("training")
    # retriver.train(dataset_path='./data/retriver_data.csv', save_path='./model/retriver-apr-7')

    # start = time.time()
    # retriver = retriver_factory('ST', load_pretrained="./model/retriver-apr-7")
    # r = retriver.retrive_most_relevant(title=parsed.title, text=parsed.body)
    # end = time.time()
    # print(f"ST retrive time {end-start:.3f}s")
    # print(r)

