import os
from dataclasses import dataclass
from dotenv import load_dotenv
from promts import *
from llm_api import *
from data_models import *
import json
from llm_core import GuardedLLMClient, STEP_POLICIES, for_extraction, get_logger

class MultiStepLLMPipeline:
    def __init__(self, extract_client : ExtractMeaningLM,
                 classifier_client : IntentClassifierLM,
                 field_extractor_client : FieldExtractorLM,
                 response_generator_client : ResponseGeneratorLM,
                 judge_client : SelfCheckLM,
                 input_path : str,
                 output_path : str,
                 logger = None):
        self.extractClient = extract_client
        self.classifierClient = classifier_client
        self.fieldExtractorClient = field_extractor_client
        self.responseGeneratorClient = response_generator_client
        self.judgeClient = judge_client
        self.input_path = input_path
        self.output_path = output_path
        self.logger = logger or get_logger()

        self.guards = {
            "extract_sense": GuardedLLMClient(extract_client, STEP_POLICIES["extract_sense"], self.logger),
            "intent":        GuardedLLMClient(classifier_client, STEP_POLICIES["intent"], self.logger),
            "final_answer":  GuardedLLMClient(response_generator_client, STEP_POLICIES["final_answer"], self.logger),
            "judge":         GuardedLLMClient(judge_client, STEP_POLICIES["judge"], self.logger),
        }

    def extract_sense(self, msg : UserMessage):
        with self.logger.context(step="extract_sense"):
            result = self.guards["extract_sense"].call([msg])

            if not result.ok:
                return None

            self.logger.info(f"sense extracted success ({len(result.content)} symbols, retries {result.attempts})")

            return result.content

    def generate_intent(self, msg : UserMessage):
        with self.logger.context(step="intent"):
            result = self.guards["intent"].call([msg])

            if not result.ok:
                return None

            answer : IntentLMAnswer = result.parsed
            self.logger.info(f"intent : {answer.intent.value} (confidence={answer.confidence}, retries: {result.attempts})")

            return answer.intent.value

    def extract_fields(self, msg : UserMessage, intent):
        with self.logger.context(step="extract_fields"):
            model_type = INTENT_MODELS[intent]

            self.fieldExtractorClient.sys_message = SystemMessage(content=COMMON_EXTRACTION_PROMPT + "\n" + EXTRACTION_PROMPT_REGISTRY[intent])
            self.fieldExtractorClient.set_response_schema_format(construct_api_payload(model_type.model_json_schema(), strict=True))
            
            guard = GuardedLLMClient(self.fieldExtractorClient, for_extraction(model_type), self.logger)
            result = guard.call([msg])

            if not result.ok:
                return None

            self.logger.info(f"fields extracted (type {model_type.__name__}, retries : {result.attempts})")

            return result.parsed

    def generate_final_answer(self, msg : UserMessage, sense : str, intent : str, sentiment : str, field_extraction):
        with self.logger.context(step="final_answer"):
            self.responseGeneratorClient.sys_message = SystemMessage(content=get_final_asnwer_prompt())

            message = UserMessage(content=build_final_answer_message(msg.content,
                                                                    intent=intent,
                                                                    sentiment=sentiment,
                                                                    sense=sense,
                                                                    field_extraction=field_extraction))
            result = self.guards["final_answer"].call([message])

            if not result.ok:
                return None

            self.logger.info(f"final answer ready ({len(result.content)} symbols, retries {result.attempts})")

            return result.content

    def generate_judge_results(self, start_msg : str, final_answer : str):
        with self.logger.context(step="judge"):
            result = self.guards["judge"].call(
                [UserMessage(content=get_judge_prompt(start_msg, final_answer))])

            if not result.ok:
                return None

            judged : JudgedLMResult = result.parsed
            self.logger.info(f"judge score: passed={judged.passed} score={judged.score})")

            return judged


    def generate_pipeline_step(self, msg : UserMessage, step : int = -1) -> PipelineLMResult | None:
        sense = self.extract_sense(msg=msg)

        if sense is None:
            self.logger.error("extract_sense step failed")
            return None

        intent = self.generate_intent(msg=msg)

        if intent is None:
            self.logger.error("intent step failed")
            return None

        field_extraction = self.extract_fields(msg=msg, intent=intent)

        if field_extraction is None:
            self.logger.error("extract_fields step failed")
            return None

        sentiment = field_extraction.sentiment.value if field_extraction.sentiment is not None else "unknown"
        final_answer = self.generate_final_answer(msg, sense, intent, sentiment, field_extraction)

        if final_answer is None:
            self.logger.error("final_answer step failed")
            return None

        judge_result = self.generate_judge_results(msg.content, final_answer)

        if judge_result is None:
            self.logger.error("judge step failed")
            return None

        return PipelineLMResult(question_index=step,
                                start_question=msg.content,
                                intent=intent,
                                field_extraction = field_extraction,
                                final_answer=final_answer,
                                judge_result=judge_result)

    def process_generation_pipeline(self, lines_to_process = -1):
        results : list[PipelineLMResult] = []
        lines_gate = lines_to_process > 0
        last_index  = 0

        self.logger.info(f"pipeline starts (run_id={self.logger.run_id}, log: {self.logger.calls_path})")

        with open(self.input_path, 'r') as f:
            for i, line in enumerate(f):

                if lines_gate and (i >= lines_to_process):
                    break

                line = line.strip()

                if not line:
                    continue

                with self.logger.context(trace_id=self.logger.new_trace(), line_idx=i):
                    self.logger.info("line %d: %s", i, line[:70])
                    answer = self.generate_pipeline_step(UserMessage(content=line), step=i)

                last_index = i

                if answer is not None:
                    results.append(answer)

        processed_indexes = {result.question_index for result in results}
        problematic_points = [i for i in range(last_index + 1) if i not in processed_indexes]

        if len(results) == 0:
            self.logger.error("pipeline ended without any results")
        else:
            with open(self.output_path, 'w') as f:
                json.dump([result.model_dump(mode="json") for result in results], f, indent=4, ensure_ascii=False)

            self.logger.info("finished: %d/%d lines -> %s", len(results), last_index + 1, self.output_path)

        if problematic_points:
            self.logger.warning("non processed lines : %s", problematic_points)

        summary = self.logger.summary()
        self.logger.info("calls: %d, success: %d (%.0f%%), summary: %s",
                         summary["total_calls"], summary["ok_calls"],
                         summary["ok_rate"] * 100, summary["by_outcome"])

        return summary

@dataclass
class LLMPipelineSettings:
    general_model_name : str
    general_model_url : str
    general_model_api_key : str
    simple_model_name : str
    simple_model_url : str
    simple_model_api_key : str

def build_llm_pipeline(settings : LLMPipelineSettings, input_path : str, output_path : str) -> MultiStepLLMPipeline: 
    
    meanining_settings = LLMSettings(model_name=settings.general_model_name,
                                     base_url=settings.general_model_url,
                                     sys_message=SystemMessage(content=SENSE_EXTRACTION_PROMPT))
    
    meaning_client = ExtractMeaningLM(api_key=settings.general_model_api_key,
                                      llm_settings=meanining_settings)
    
    intent_client = IntentClassifierLM(api_key=settings.simple_model_api_key,
                                       model_name=settings.simple_model_name,
                                       sys_message=SystemMessage(content=INTENT_PROMPT),
                                       payload_schema=INTENT_PAYLOAD_SCHEMA)
    
    field_extractor_settings = LLMSettings(model_name=settings.simple_model_name,
                                           temperature=.01)
    
    field_extractor_client = FieldExtractorLM(api_key=settings.simple_model_api_key,
                                              llm_settings=field_extractor_settings)
    
    response_generator_settings = LLMSettings(model_name=settings.general_model_name, 
                                              base_url=settings.general_model_url)
    
    response_generator_client = ResponseGeneratorLM(api_key=settings.general_model_api_key,
                                                    llm_settings=response_generator_settings)
    
    judge_settings = LLMSettings(model_name=settings.general_model_name,
                                 base_url=settings.general_model_url,
                                 payload_schema=construct_api_payload(JudgedLMResult.model_json_schema()),
                                 sys_message=SystemMessage(content=SELF_CHECK_PROMPT),
                                 temperature=.01)
    
    judge_client = SelfCheckLM(api_key=settings.general_model_api_key,
                               llm_settings=judge_settings)
    
    return MultiStepLLMPipeline(extract_client=meaning_client,
                                classifier_client=intent_client,
                                field_extractor_client=field_extractor_client,
                                response_generator_client=response_generator_client,
                                judge_client=judge_client,
                                input_path=input_path,
                                output_path=output_path)