from transformers import AutoTokenizer
import torch as torch
from openai import OpenAI,Timeout
from dataclasses import dataclass, field
from data_models import SystemMessage, Message, UserMessage

DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)
MAX_RETRIES = 2

@dataclass
class LLMSettings:
    model_name : str
    base_url : str | None = None
    payload_schema : dict | None = None
    payload_type : type | None = None
    sys_message : SystemMessage | None = None
    max_tokens : int = 1024
    temperature : float = .95
    top_p : float = -1
    max_retries : int = MAX_RETRIES
    timeout: Timeout = field(default_factory=lambda: DEFAULT_TIMEOUT)
    
class LLMClient:
    def __init__(self, api_key, llm_settings : LLMSettings):
        self.payload_settings = llm_settings
        self.client = OpenAI(api_key=api_key,
                             base_url=llm_settings.base_url,
                             timeout=llm_settings.timeout,
                             max_retries=llm_settings.max_retries)
        self.apply_settings(llm_settings)
        print(f"Init llm client with model {llm_settings.model_name}")

    def apply_settings(self, llm_settings : LLMSettings):
        self.sys_message = llm_settings.sys_message
        self.payload_type = llm_settings.payload_type
        self.construct_payload(llm_settings)
    
    def construct_payload(self, llm_settings : LLMSettings):        
        payload = {
            "model": llm_settings.model_name,
            "temperature": llm_settings.temperature,
            "max_tokens": llm_settings.max_tokens,
        }

        if llm_settings.payload_schema is not None:
            payload["response_format"] = llm_settings.payload_schema
        
        if llm_settings.top_p > 0:
            payload["top_p"] = llm_settings.top_p
        
        self.payload = payload
    
    def set_response_schema_format(self, response_schema : str):
        self.payload["response_format"] = response_schema
    
    def generate(self, message_list : list[Message]):
        payload = self.payload
    
        messages = [msg.convert() for msg in message_list]
        
        if (self.sys_message is not None):
            messages.insert(0, self.sys_message.convert())
        
        payload["messages"] = messages  # reuse messages each time u request generation (no structures/value types in python is sick)

        result = self.client.chat.completions.create(**payload).choices[0].message.content
        
        return result
    
class IntentClassifierLM:
    def __init__(self, api_key : str, model_name : str, sys_message : SystemMessage, payload_schema : dict, temperature = 0.01):
        self.api_key = api_key
        self.model_name = model_name
        self.sys_message = sys_message
        self.payload_schema = payload_schema
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        self.construct_payload()
        print(f"Init llm intent classifier with model {self.model_name}")
        
    def construct_payload(self):
        self.payload = {
            "model": self.model_name,
            "response_format" : self.payload_schema,
            "temperature": self.temperature,
        }
    
    def generate_classification(self, msg : Message):
        payload = self.payload
        
        messages = [self.sys_message.convert()]
        messages.append(msg.convert())
        
        payload["messages"] = messages
        
        result = self.client.chat.completions.create(**payload).choices[0].message.content
        
        return result
    
class ExtractMeaningLM(LLMClient):
    pass

class FieldExtractorLM(LLMClient):
    pass

class ResponseGeneratorLM(LLMClient):
    pass

class SelfCheckLM(LLMClient):
    pass

