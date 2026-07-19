from dataclasses import dataclass
from pydantic import Field, BaseModel
import json
import enum
from typing import Literal, Any

class SentimentResult(enum.Enum):
    positive    =   "positive"
    negative    =   "negative"
    neutral     =   "neutral"
        
class IntentResult(enum.Enum):
    support             =   "support"
    feedback            =   "feedback"
    complaint           =   "complaint"
    sales               =   "sales"
    general_question    =   "general_question"
        
class Urgency(enum.Enum):
    low     =   "low"
    medium  =   "medium"
    high    =   "high"
            
class FailureKind(enum.Enum):
    empty           =   "empty"
    invalid_json    =   "invalid_json"
    schema          =   "schema"
    too_long        =   "too_long"
    truncateed      =   "truncated"
    api_error       =   "api_error"
    fatal           =   "fatal"

class Criticality(enum.Enum):
    required    =   "required"
    degradable  =   "degradable"
    optional    =   "optional"

@dataclass
class Message:
    role : str
    content : str | None = None

    def convert(self):
        return self.__dict__

@dataclass
class UserMessage(Message):
    role : str = "user"
    
@dataclass
class SystemMessage(Message):
    role : str = "system"

@dataclass
class LLMSummarizationAnswer:
    summary : str
    key_thoughts : list[str]
    response : str
    
    @staticmethod
    def validate_answer_json(llm_answer : str):
        try:
            data = json.loads(llm_answer)
            
            if not isinstance(data, dict):
                return False
            
            answer = LLMSummarizationAnswer(**data)
            
            if (len(answer.key_thoughts) != 3):
                return False
            
            if (len(answer.summary) > 50 or len(answer.summary) < 25):
                return False
            
            return True
        except (json.JSONDecodeError, TypeError):
            return False

class LLMSummarizationAnswer_LLM_DAY3(BaseModel):
    summary : str = Field(min_length=25, max_length=100)
    category :str = Field(min_length=1, max_length=20)
    sentiment : SentimentResult = Field()
    key_points : list[str] = Field(min_length=3, max_length=3)
    final_answer : str = Field(min_length=1, max_length=75)
    
    @staticmethod
    def validate_answer_json(llm_answer : str):
        try:
            data = json.loads(llm_answer)
            answer = LLMSummarizationAnswer_LLM_DAY3(**data)
            return True
        except Exception as e:
            print("Validation error:", e)
            return False

class IntentLMAnswer(BaseModel):
    intent : IntentResult
    confidence : int = Field(ge=1, le=5)
    reason : str = Field(min_length=10, max_length=80)
    
    @staticmethod
    def validate_answer_json(llm_answer : str):
        try:
            data = json.loads(llm_answer)
            answer = IntentLMAnswer(**data)
            return True
        except Exception as e:
            print("Validation error:", e)
            return False

class PipelineLMResult(BaseModel):
    question_index : int
    start_question : str
    intent : str
    field_extraction : Any
    final_answer : str
    judge_result : str

class JudgedLMResult(BaseModel):
    passed : bool
    score : int = Field(ge=1, le=10)
    issues : list[str] = []

class CommonFields(BaseModel):
    summary : str | None = None
    sentiment : SentimentResult | None = None
    urgency : Urgency | None = None
    language : str  | None = None
    intent : IntentResult | None = None
    
class SupportFields(CommonFields):
    problem : str | None = None
    product : str | None = None
    error_code : str | None = None
    
class SalesFields(CommonFields):
    interester_product : str | None = None
    budget : str | None = None
    items_count : str | None = None
    
class ComplaintFields(CommonFields):
    complaint_reason : str | None = None
    requested_resolution : str | None = None
    
class FeedbackFields(CommonFields):
    feedback_type: Literal["positive", "negative", "suggestion", "mixed"] | None = None
    subject: str | None = None
    feature: str | None = None
    suggestion: str | None = None
    
class GeneralQuestionFields(CommonFields):
    topic: str | None = None
    question_type: Literal["how_to", "information", "comparison", "definition", "other"] | None = None
    entities: list[str] | None = None

@dataclass
class LLMCallResult:
    ok: bool
    content: str | None
    parsed: BaseModel | None
    failure: FailureKind | None
    attempts: int
    fallback_level: int
    finish_reason: str | None
    usage: dict | None
    latency_ms: int
    raw_error: str | None

@dataclass
class StepPolicy:
    name: str
    model: BaseModel | None          # схема валидации
    soft_max_chars: int | None       # предупреждение
    hard_max_chars: int | None       # отбраковка
    min_chars: int
    criticality: Criticality         # REQUIRED | DEGRADABLE | OPTIONAL
    fallback_chain: list
    retry_budget: int
    temperature_ladder: list[float]  # напр. [0.95, 0.3, 0.0]

    
INTENT_MODELS = {
    "support": SupportFields,
    "sales": SalesFields,
    "complaint": ComplaintFields,
    "feedback": FeedbackFields,
    "general_question": GeneralQuestionFields
}

def construct_api_payload(payload_schema : str):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": payload_schema
        }
    }
    
