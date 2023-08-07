import os, json, string, io
import PyPDF2
from typing import Callable
from fastapi import UploadFile
from ..base import BaseTool

class FileProcessTool(BaseTool):
    name: str = "file_process"
    description: str = "Tool for process files to improve the discussion"
    user_description: str = "You can enable this to upload some pdf files for discussion"
    usable_by_bot = False

    def __init__(self, func: Callable=None, **kwargs):
        self.word_limit = kwargs.get("pdf_word_limit", 1000)
        self.current_user_files = {}
        OnStartUp = kwargs.get("OnStartUp")
        OnStartUpMsgEnd = kwargs.get("OnStartUpMsgEnd")
        OnUserMsgReceived = kwargs.get("OnUserMsgReceived")
        OnResponseEnd = kwargs.get("OnResponseEnd")
        OnUserDisconnected = kwargs.get("OnUserDisconnected")
        
        OnUserMsgReceived += self.OnUserMsgReceived

        OnUserDisconnected += self.OnUserDisconnected

        super().__init__(None)
    
    async def handle_file_upload(self, user_id, file: UploadFile):
        try:
            file_contents = await file.read() # binary value of the file
            pdfReader = PyPDF2.PdfReader(io.BytesIO(file_contents))

            # Extract text from each page
            all_text = ""
            for page in range(len(pdfReader.pages)):
                pageObj = pdfReader.pages[page]
                text = pageObj.extract_text()
                all_text += text + "\n"
                if len(all_text.split(" ")) > self.word_limit:
                    break
            self.current_user_files[user_id] = all_text
            print(f"Upload file: user id: {user_id}, file name: {file.filename}")
            return {"user_id": user_id, "filename": file.filename, "status": "success"}
        except Exception as e:
            print(e)
            return {"user_id": user_id, "status": "failed", "detail": str(e)}
    
    def OnUserMsgReceived(self, **kwargs):
        user_assistants = kwargs.get("user_assistants", None)
        user_id = kwargs.get("user_id")
        if (user_assistants is None) or (user_id not in self.current_user_files):
            return
        for user_assistant_update in user_assistants:
            user_assistant_update(self.current_user_files[user_id])

    def OnUserDisconnected(self, **kwargs):
        user_id = kwargs.get("user_id")
        self.remove_user_files(user_id)
    
    def remove_user_files(self, user_id):
        self.current_user_files.pop(user_id, "")

    def _run(self):
        return None