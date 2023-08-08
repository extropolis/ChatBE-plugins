import os, json, string, io
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import PyPDF2
from typing import Callable, List, Dict
from fastapi import UploadFile
from ..base import BaseTool

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
    user_description: str = "You can enable this to upload some pdf files for discussion"
    usable_by_bot = False

    def __init__(self, func: Callable=None, **kwargs):
        self.word_limit = kwargs.get("file_word_limit", 1000)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"] # for langchain compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.word_limit, chunk_overlap=int(self.word_limit / 5))

        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_API_ENV"])

        self.current_user_files: Dict[str, Pinecone] = {}
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
        return {"user_id": user_id, "url": url, "status": "success"}
    
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
            self.current_user_files[user_id] = Pinecone.from_documents(splitted_doc, self.embeddings, index_name="filedix")
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
        
        docs = self.current_user_files[user_id].as_retriever(search_kwargs={"k": 5}).get_relevant_documents(user_msg)
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
            self.current_user_files[user_id].delete(delete_all=True) # TODO: this deletes everything, we need to properly handle it for different users...

    def _run(self):
        return None