from typing import Callable
from collections import defaultdict
from datetime import date
from google.cloud.firestore_v1.base_client import BaseClient
from ..base import BaseTool
import openai
import os, dotenv, json
import asyncio

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_KEY"]

DEFAULT_USER_FACT = {
    "name": None,
    "age": None,
    "gender": None,
    "home_city": None,
    "occupation": None,
    "employer": None,
    "spouce": None,
    "friends": [],
    "kids": [],
    "interests": [],
}


class Memory:
    def __init__(self, db: BaseClient=None, short_memory_length: int=10, model="gpt-3.5-turbo-16k"):
        '''
            short_memory_length: length of conversation considered for the short memory. This is the number of user messages. if short_memory_length=1, the short memory will actually include at least 3 messages (1 initial message from AI, 1 from user, 1 from the AI assistant.)
        '''
        if db == None:
            raise Exception("database is None when initializing memory")
        self.memory = db.collection("memory")
        self.active_user_sessions = defaultdict(list)
        self.user_short_memory = defaultdict(list)
        self.short_mem_len = short_memory_length
        self.model = model
        if self.model.endswith("16k"):
            self.threshold = 14000
        elif self.model.endswith("32k"):
            self.threshold = 28000
        else:
            self.threshold = 2000

    def store_user_facts(self, user_id, facts):
        user_fact_ref = self.memory.document(user_id)
        user_fact_doc = user_fact_ref.get()
        if user_fact_doc and user_fact_doc.exists:
            new_user_facts = user_fact_doc.to_dict()
            for k, v in facts.items():
                if v is not None and isinstance(v, (str, int, dict)):
                    new_user_facts[k] = v
                elif isinstance(v, list):
                    new_user_facts[k].extend(v)
                    # remove duplicates
                    new_user_facts[k] = list(set(new_user_facts[k]))
            user_fact_ref.update(new_user_facts)
            print(f"GPT generated facts:\n {json.dumps(facts, indent=True)}")
            print(f"Updated user facts:\n {json.dumps(new_user_facts, indent=True)}")
        else:
            user_fact_ref.set(facts)
            print(facts)
    
    def get_memory(self, user_id):
        user_fact_ref = self.memory.document(user_id)
        user_fact_doc = user_fact_ref.get()
        if user_fact_doc and user_fact_doc.exists:
            return user_fact_doc.to_dict()
        else:
            facts = DEFAULT_USER_FACT.copy()
            return facts
    
    def __trim_conversation(self, conversations):
        total_len = 0
        for i in range(len(conversations) - 1, -1, -1):
            total_len += len(conversations[i]["content"].split(" ")) + 1 # to also account for the role token
            if total_len > self.threshold:
                print(f"Session length: {total_len}")
                return conversations[i:]
        print(f"Session length: {total_len}")
        return conversations
    
    async def generate_user_facts(self, user_id):
        user_facts = self.get_memory(user_id)
        new_message = """Based on the conversation history, update the following information about the 'user' in JSON format. Output only the completed JSON object. Write 'null' if the information does not exist. You should focus on what the user says. The assistant messages should only be taken as references and not factual about the user.
        Format:
        {
            "name": <string>,
            "age": <int>,
            "gender": <string>,
            "home_city": <string>,
            "occupation": <string>,
            "employer": <string>,
            "spouce": <string>,
            "friends": [],
            "kids": [],
            "interests": [],
        }"""
        new_message += f"""
        Known user facts:
        {json.dumps(user_facts, indent=True)}
        """

        # This is just for safety, in case we have too large a short_mem_len set
        short_conversation = self.__trim_conversation(self.user_short_memory[user_id])

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages= short_conversation + [{"role": "user", "content": new_message}],
            temperature=0
        )
        
        try:
            content = response['choices'][0]['message']['content']
            user_facts = json.loads(content)
            # user_facts["last_conversation"] = date.today().isoformat()
        except Exception as e:
            print(e)

        self.store_user_facts(user_id, user_facts)

    async def generate_follow_up(self, user_id):
        print("GENERATING FOLLOW UPS")
        user_facts = self.get_memory(user_id)
        new_message = """Summarize the conversation and generate some follow up ideas to start the next conversation. Output only the complete JSON object. The summarization and each follow up idea should not exceed 100 words.
        Format:
        {
            "summary": <string>,
            "follow_up": [],
        }"""
        user_msg_count = sum(1 for d in self.active_user_sessions[user_id] if d["role"] == "user")
        if user_msg_count < 1:
            print("No user interaction in the conversation, should not overwrite the conversation history")
            return

        conversations = self.__trim_conversation(self.active_user_sessions[user_id])

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=conversations + [{"role": "user", "content": new_message}],
            # temperature=0
        )

        try:
            content = response['choices'][0]['message']['content']
            print(content)
            conversation_summary = json.loads(content)
            conversation_summary["timestamp"] = date.today().isoformat()
            print(conversation_summary)
            user_facts["last_conversation"] = conversation_summary
            # user_facts["last_conversation"] = date.today().isoformat()
        except Exception as e:
            print(e)
        self.store_user_facts(user_id, user_facts)

    async def update_memory(self, user_id, user_session_finished=False):
        if len(self.active_user_sessions[user_id]) == 0:
            print("nothing to update")
            return
        user_msg_count = sum(1 for d in self.user_short_memory[user_id] if d["role"] == "user")
        if user_msg_count < len(self.user_short_memory[user_id]) // 2 or len(self.user_short_memory[user_id]) == 0:
            # user has talked too little, should not update the memory
            print("User has talked too little, do not update short term memory")
        else:
            await self.generate_user_facts(user_id)
        self.user_short_memory[user_id].clear()

        if user_session_finished:
            await self.generate_follow_up(user_id)
            self.active_user_sessions[user_id].clear()
    
    def clear_memory(self, user_id):
        user_fact_ref = self.memory.document(user_id)
        user_fact_doc = user_fact_ref.get()
        if user_fact_doc and user_fact_doc.exists:
            user_fact_ref.delete()
        else:
            return

    async def update_user_session(self, user_id, message):
        assert isinstance(message, dict) and "role" in message and "content" in message, "message must be a dictionary containing role and content fields"
        assert message["role"] in ["user", "assistant"], "message role must be one of user, assistant"
        self.active_user_sessions[user_id].append(message)
        self.user_short_memory[user_id].append(message)

        user_msg_count = sum(1 for d in self.user_short_memory[user_id] if d["role"] == "user")
        if user_msg_count >= self.short_mem_len and self.user_short_memory[user_id][-1]["role"] == "assistant":
            # Hacky way of checking the messages, need a better approach
            await self.update_memory(user_id)
            print("memory about user updated")
        
class MemoryTool(BaseTool):
    name: str = "memory"
    description: str = "Tool for memorizing facts about users. Contains long term and short term memory about users. When this tool is enabled, initial prompts for AI assistants will be updated according to known user facts. However, this tool itself should not be used by the AI assistants."
    user_description: str = "You can enable this to get a better continuation between different conversations."
    usable_by_bot: bool = False
    def __init__(self, func: Callable=None, **kwargs) -> None:
        db = kwargs.get("db", None)
        short_memory_length = kwargs.get("short_memory_length", 5)
        memory_model = kwargs.get("memory_model", "gpt-3.5-turbo-16k")
        self.memory = Memory(db=db, short_memory_length=short_memory_length, model=memory_model)
        # All the handlers must have been correctly setup, otherwise Memory is no use, 
        # so if there is any error, we must raise
        OnStartUp = kwargs.get("OnStartUp")
        OnStartUpMsgEnd = kwargs.get("OnStartUpMsgEnd")
        OnUserMsgReceived = kwargs.get("OnUserMsgReceived")
        OnResponseEnd = kwargs.get("OnResponseEnd")
        OnUserDisconnected = kwargs.get("OnUserDisconnected")
        OnStartUp += self.OnStartUp
        OnStartUpMsgEnd += self.OnStartUpMsgEnd
        OnUserMsgReceived += self.OnUserMsgReceived
        OnResponseEnd += self.OnResponseEnd
        OnUserDisconnected += self.OnUserDisconnected

        super().__init__(None)

    def OnStartUp(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        if user_tool_settings[self.name]:
            mem = self.get_memory(user_id)
            for k, v in mem.items():
                user_info[k] = v
    
    def OnStartUpMsgEnd(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        message = kwargs.get("message")
        if user_tool_settings[self.name]:
            asyncio.create_task(self.update_user_session(user_id, message))

    def OnUserMsgReceived(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        message = kwargs.get("message")
        if user_tool_settings[self.name]:
            asyncio.create_task(self.update_user_session(user_id, message))
            mem = self.get_memory(user_id)
            for k, v in mem.items():
                user_info[k] = v

    def OnResponseEnd(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        message = kwargs.get("message")
        if user_tool_settings[self.name]:
            asyncio.create_task(self.update_user_session(user_id, message))

    def OnUserDisconnected(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        if user_tool_settings[self.name]:
            asyncio.create_task(self.update_memory(user_id, True))
    
    def get_memory(self, user_id):
        return self.memory.get_memory(user_id)

    async def update_memory(self, user_id, user_session_finished=False):
        await self.memory.update_memory(user_id, user_session_finished)
    
    async def clear_memory(self, user_id):
        self.memory.clear_memory(user_id)

    async def update_user_session(self, user_id, message):
        await self.memory.update_user_session(user_id, message)

    def _run(self, query: str):
        return None

