from pydantic import BaseModel
from typing import Optional, Any
from fastapi import UploadFile
import requests
from urllib.parse import urlparse
import hashlib
from .file_processors import ALL_PROCESSORS

default_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.1901.188 Safari/537.36 Edg/115.0.1901.188",
    "Upgrade-Insecure-Requests": "1",
}

class CustomFileProcessor(BaseModel):
    file: Optional[UploadFile] = None
    file_url: Optional[str] = None
    file_name: Optional[str] = ""
    file_size: Optional[int] = 0
    file_sha1: Optional[str] = ""
    content_type: Optional[str] = ""
    content: Optional[Any] = None
    meta_data: Optional[dict] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def load_file(self):
        if self.file:
            self.file_name = self.file.filename
            self.content_type = self.file.headers["content-type"].split(";")
            self.content = await self.file.read()
            try:
                self.file_size = self.file.file.tell()
            except:
                self.file_size = -1
        elif self.file_url:
            parsed_url = urlparse(self.file_url)
            path = parsed_url.path
            self.file_name = path.split("/")[-1]

            getReq = requests.get(self.file_url, headers=default_headers)
            self.content_type = getReq.headers["Content-Type"].split(";")
            self.content = getReq.content
            try:
                self.file_size = int(getReq.headers["Content-length"])
            except:
                self.file_size = -1
        else:
            raise Exception("Empty file and url. Make sure you have uploaded at least one file")
        if self.file_name.endswith(".csv"):
            self.content_type = ["text/csv"]
        if self.file_name.endswith((".xlsx", ".xls")):
            self.content_type = ["application/vnd.ms-excel"]
        self.compute_sha1()
        self.meta_data = {
            "name": self.file_name,
            "source": "Upload" if self.file else self.file_url,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "file_sha1": self.file_sha1
        }
        print(self.meta_data)
    
    def compute_sha1(self):
        self.file_sha1 = hashlib.sha1(self.content).hexdigest()

    def generate_docs(self):
        for t in self.content_type:
            if t in ALL_PROCESSORS:
                return ALL_PROCESSORS[t](self.content, self.meta_data)
        raise Exception("Unsupported file type by url. Make sure the content-type provided is one of application/json, application/pdf, text/*")
