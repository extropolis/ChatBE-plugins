import os, json, string, io
import requests
from urllib.parse import urlparse
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import PyPDF2
from typing import Callable, List, Dict
from fastapi import UploadFile
from ..base import BaseTool

default_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.1901.188 Safari/537.36 Edg/115.0.1901.188",
    "Upgrade-Insecure-Requests": "1",
}

def process_txt_file(file_contents: bytes, meta_data: dict) -> List[Document]:
    all_text = file_contents.decode("utf-8", errors="ignore")
    return [Document(page_content=all_text, metadata=meta_data)]

def process_pdf_file(file_contents: bytes, meta_data: dict) -> List[Document]:
    pdfReader = PyPDF2.PdfReader(io.BytesIO(file_contents))

    # Extract text from each page
    doc_list = []
    for page in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[page]
        text = pageObj.extract_text()
        curr_doc = Document(page_content=text, metadata={"source": meta_data["name"], "page": page})
        doc_list.append(curr_doc)
    return doc_list

class FileProcessTool(BaseTool):
    name: str = "file_process"
    description: str = "Tool for process files to improve the discussion"
    user_description: str = "You can enable this to upload some files or submit a link for discussion. It is suggested that you remove any uploaded files after you are done, to have a better conversation flow."
    usable_by_bot = False

    def __init__(self, func: Callable=None, **kwargs):
        self.word_limit = kwargs.get("file_word_limit", 1000)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"] # for langchain compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.word_limit, chunk_overlap=int(self.word_limit / 5))

        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_API_ENV"])
        self.pinecone_db = Pinecone(index=pinecone.Index("fileidx"), embedding_function=self.embeddings.embed_query, text_key="text")

        self.current_user_files: Dict[str, List[str]] = {} # this dictionary maps user id to a list of document ids recorded in the database
        OnStartUp = kwargs.get("OnStartUp")
        OnStartUpMsgEnd = kwargs.get("OnStartUpMsgEnd")
        OnUserMsgReceived = kwargs.get("OnUserMsgReceived")
        OnResponseEnd = kwargs.get("OnResponseEnd")
        OnUserDisconnected = kwargs.get("OnUserDisconnected")
        
        OnUserMsgReceived += self.OnUserMsgReceived
        OnUserDisconnected += self.OnUserDisconnected

        super().__init__(None)
    
    def handle_url(self, user_id, url):
        print(f"user id: {user_id}, url: {url}")
        try:
            url_header = requests.head(url, headers=default_headers)
            content_types = url_header.headers["Content-Type"].split(";")
            content_type = None
            for t in content_types:
                if t in ["application/pdf", "application/json"]:
                    content_type = t
                    break
                ts = t.split("/")
                if ts[0] == "text" :
                    content_type = "text"
                    break
            parsed_url = urlparse(url)
            path = parsed_url.path
            base_name = path.split("/")[-1]
            if content_type == "application/pdf":
                print("pdf file")
                file_content = requests.get(url, headers=default_headers).content
                all_text = process_pdf_file(file_content, {"source": url, "name": base_name})
            elif content_type == "text" or content_type == "application/json":
                print("txt file")
                file_content = requests.get(url, headers=default_headers).content
                all_text = process_txt_file(file_content, {"source": url, "name": base_name})
            else:
                raise Exception("Unsupported file type by url. Make sure the content-type provided is one of application/json, application/pdf, text/*")
            splitted_doc = self.text_splitter.split_documents(all_text)
            
            self.remove_user_files(user_id)
            self.current_user_files[user_id] = self.pinecone_db.add_documents(splitted_doc, namespace=f"{user_id}-files")
            return {"user_id": user_id, "url": url, "status": "success"}
        except Exception as e:
            print(e)
            return {"user_id": user_id, "status": "failed", "detail": str(e)}
    
    async def handle_file_upload(self, user_id, file: UploadFile):
        try:
            print(f"Upload file: user id: {user_id}, file name: {file.filename}, file header: {file.headers}")
            meta_data = dict(file.headers).copy()
            meta_data["name"] = file.filename
            file_contents = await file.read() # binary value of the file
            if file.headers["content-type"] == "text/plain" \
                or file.headers["content-type"] == "application/json" \
                or file.headers["content-type"].split("/")[0] == "text": # txt file, json file or scripts
                all_text = process_txt_file(file_contents, meta_data) 
            elif file.headers["content-type"] == "application/pdf": # pdf file
                all_text = process_pdf_file(file_contents, meta_data)
            else:
                raise Exception("Unsupported file type. We accept plain text, json or pdf files")
            splitted_doc = self.text_splitter.split_documents(all_text)
            
            self.remove_user_files(user_id)
            self.current_user_files[user_id] = self.pinecone_db.add_documents(splitted_doc, namespace=f"{user_id}-files")
            return {"user_id": user_id, "filename": file.filename, "status": "success"}
        except Exception as e:
            print(e)
            return {"user_id": user_id, "status": "failed", "detail": str(e)}
    
    def OnUserMsgReceived(self, **kwargs):
        user_assistants = kwargs.get("user_assistants", None)
        user_msg = kwargs.get("message")["content"]
        user_id = kwargs.get("user_id")
        if (user_assistants is None) or (user_id not in self.current_user_files):
            return
        
        docs = self.pinecone_db.as_retriever(search_kwargs={"k": 5, "namespace": f"{user_id}-files"}).get_relevant_documents(user_msg)
        all_text = "\n\n".join([doc.page_content for doc in docs])
        print(len(docs))
        print(all_text)
        for user_assistant_update in user_assistants:
            user_assistant_update(all_text)

    def OnUserDisconnected(self, **kwargs):
        user_id = kwargs.get("user_id")
        self.remove_user_files(user_id)
    
    def remove_user_files(self, user_id):
        if user_id in self.current_user_files and self.current_user_files[user_id] is not None:
            self.pinecone_db.delete(ids=self.current_user_files[user_id], namespace=f"{user_id}-files") 

    def _run(self):
        return None