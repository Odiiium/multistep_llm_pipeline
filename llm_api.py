from openai import OpenAI,Timeout
from dataclasses import dataclass, field
from data_models import SystemMessage, Message, UserMessage, RawResponse

DEFAULT_TIMEOUT = Timeout(connect=5.0, read=60, write=60, pool=60)
MAX_RETRIES = 0

def to_raw_response(completion) -> RawResponse:
    choice = completion.choices[0] if completion.choices else None
    message = getattr(choice, "message", None) if choice else None
    usage = getattr(completion, "usage", None)

    return RawResponse(
        content=getattr(message, "content", None),
        finish_reason=getattr(choice, "finish_reason", None) if choice else None,
        usage=usage.model_dump() if hasattr(usage, "model_dump") else usage,
        refusal=getattr(message, "refusal", None),
    )


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
    
    def build_messages(self, message_list : list[Message]):
        messages = [msg.convert() for msg in message_list]

        if (self.sys_message is not None):
            messages.insert(0, self.sys_message.convert())

        return messages

    def generate_raw(self, message_list : list[Message], temperature : float | None = None) -> RawResponse:
        payload = dict(self.payload)
        payload["messages"] = self.build_messages(message_list)

        if temperature is not None:
            payload["temperature"] = temperature

        return to_raw_response(self.client.chat.completions.create(**payload))

    def generate(self, message_list : list[Message]):
        return self.generate_raw(message_list).content

class IntentClassifierLM:
    def __init__(self, api_key : str, model_name : str, sys_message : SystemMessage, payload_schema : dict, temperature = 0.01, base_url : str | None = None):
        self.api_key = api_key
        self.model_name = model_name
        self.sys_message = sys_message
        self.payload_schema = payload_schema
        self.temperature = temperature
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key,
                             base_url=base_url,
                             timeout=DEFAULT_TIMEOUT,
                             max_retries=MAX_RETRIES)
        self.construct_payload()
        print(f"Init llm intent classifier with model {self.model_name}")
        
    def construct_payload(self):
        self.payload = {
            "model": self.model_name,
            "response_format" : self.payload_schema,
            "temperature": self.temperature,
        }
    
    def generate_raw(self, message_list : list[Message], temperature : float | None = None) -> RawResponse:
        payload = dict(self.payload)
        payload["messages"] = [self.sys_message.convert()] + [m.convert() for m in message_list]

        if temperature is not None:
            payload["temperature"] = temperature

        return to_raw_response(self.client.chat.completions.create(**payload))

    def generate_classification(self, msg : Message):
        return self.generate_raw([msg]).content

class ExtractMeaningLM(LLMClient):
    pass

class FieldExtractorLM(LLMClient):
    pass

class ResponseGeneratorLM(LLMClient):
    pass

class SelfCheckLM(LLMClient):
    pass

