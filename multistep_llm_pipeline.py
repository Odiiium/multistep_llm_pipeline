import os
from dataclasses import dataclass
from dotenv import load_dotenv
from promts import *
from llm_api import *
from data_models import *
import json
from llm_core import GuardedLLMClient, STEP_POLICIES, for_extraction, get_logger
from llm_core.fallback import classify_intent_offline, condense_message, empty_fields_for

DETERMINISTIC_LEVEL = 3
SKIPPED_LEVEL = 4

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

    def extract_sense(self, msg : UserMessage) -> LLMCallResult:
        with self.logger.context(step="extract_sense"):
            result = self.guards["extract_sense"].call([msg])

            if result.ok:
                self.logger.info("sense extracted (%d chars, attempts %d)", len(result.content), result.attempts)

            return result

    def generate_intent(self, msg : UserMessage) -> LLMCallResult:
        with self.logger.context(step="intent"):
            result = self.guards["intent"].call([msg])

            if result.ok:
                answer : IntentLMAnswer = result.parsed
                self.logger.info("intent %s (confidence=%d, attempts %d)",
                                 answer.intent.value, answer.confidence, result.attempts)

            return result

    def extract_fields(self, msg : UserMessage, intent) -> LLMCallResult:
        with self.logger.context(step="extract_fields"):
            model_type = INTENT_MODELS[intent]

            self.fieldExtractorClient.sys_message = SystemMessage(content=COMMON_EXTRACTION_PROMPT + "\n" + EXTRACTION_PROMPT_REGISTRY[intent])
            self.fieldExtractorClient.set_response_schema_format(construct_api_payload(model_type.model_json_schema(), strict=True))

            guard = GuardedLLMClient(self.fieldExtractorClient, for_extraction(model_type), self.logger)
            result = guard.call([msg])

            if result.ok:
                self.logger.info("fields extracted (%s, attempts %d)", model_type.__name__, result.attempts)

            return result

    def generate_final_answer(self, msg : UserMessage, sense : str, intent : str, sentiment : str, field_extraction) -> LLMCallResult:
        with self.logger.context(step="final_answer"):
            self.responseGeneratorClient.sys_message = SystemMessage(content=get_final_asnwer_prompt())

            message = UserMessage(content=build_final_answer_message(msg.content,
                                                                    intent=intent,
                                                                    sentiment=sentiment,
                                                                    sense=sense,
                                                                    field_extraction=field_extraction))
            result = self.guards["final_answer"].call([message])

            if result.ok:
                self.logger.info("final answer ready (%d chars, attempts %d)", len(result.content), result.attempts)

            return result

    def generate_judge_results(self, start_msg : str, final_answer : str) -> LLMCallResult:
        with self.logger.context(step="judge"):
            result = self.guards["judge"].call(
                [UserMessage(content=get_judge_prompt(start_msg, final_answer))])

            if result.ok:
                judged : JudgedLMResult = result.parsed
                self.logger.info("judge passed=%s score=%d", judged.passed, judged.score)

            return result

    def mark_degraded(self, name : str, result : LLMCallResult, degraded : list, levels : dict, level : int):
        degraded.append(name)
        levels[name] = level

        failure = result.failure.value if result.failure else "unknown"
        self.logger.warning("%s degraded to level %d (%s): %s", name, level, failure, result.detail)


    def generate_pipeline_step(self, msg : UserMessage, step : int = -1) -> PipelineLMResult | None:
        degraded : list[str] = []
        levels : dict[str, int] = {}

        sense_result = self.extract_sense(msg=msg)

        if sense_result.ok:
            sense = sense_result.content
            levels["extract_sense"] = sense_result.fallback_level
        else:
            sense = condense_message(msg.content)
            self.mark_degraded("extract_sense", sense_result, degraded, levels, DETERMINISTIC_LEVEL)

        intent_result = self.generate_intent(msg=msg)

        if intent_result.ok:
            intent = intent_result.parsed.intent.value
            levels["intent"] = intent_result.fallback_level
        else:
            intent = classify_intent_offline(msg.content)
            self.mark_degraded("intent", intent_result, degraded, levels, DETERMINISTIC_LEVEL)
            self.logger.warning("intent resolved offline as %s", intent)

        fields_result = self.extract_fields(msg=msg, intent=intent)

        if fields_result.ok:
            field_extraction = fields_result.parsed
            levels["extract_fields"] = fields_result.fallback_level
        else:
            field_extraction = empty_fields_for(intent)
            self.mark_degraded("extract_fields", fields_result, degraded, levels, DETERMINISTIC_LEVEL)

        sentiment = field_extraction.sentiment.value if field_extraction.sentiment is not None else "unknown"
        answer_result = self.generate_final_answer(msg, sense, intent, sentiment, field_extraction)

        if not answer_result.ok:
            self.logger.error("final_answer is required and produced nothing, dropping line %d", step)
            return None

        levels["final_answer"] = answer_result.fallback_level

        judge_result = self.generate_judge_results(msg.content, answer_result.content)

        if judge_result.ok:
            judged = judge_result.parsed
            levels["judge"] = judge_result.fallback_level
        else:
            judged = None
            self.mark_degraded("judge", judge_result, degraded, levels, SKIPPED_LEVEL)

        if degraded:
            self.logger.warning("line %d completed with degraded steps: %s", step, degraded)

        return PipelineLMResult(question_index=step,
                                start_question=msg.content,
                                intent=intent,
                                field_extraction = field_extraction,
                                final_answer=answer_result.content,
                                judge_result=judged,
                                degraded_steps=degraded,
                                fallback_levels=levels)

    @property
    def partial_path(self) -> str:
        return os.path.splitext(self.output_path)[0] + "_partial.jsonl"

    def read_input_lines(self, lines_to_process : int) -> list[tuple[int, str]]:
        lines_gate = lines_to_process > 0
        selected = []

        with open(self.input_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                if lines_gate and (i >= lines_to_process):
                    break

                line = line.strip()

                if line:
                    selected.append((i, line))

        return selected

    def append_partial(self, handle, result : PipelineLMResult):
        handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def write_output(self, results : list[PipelineLMResult]):
        with open(self.output_path, 'w', encoding="utf-8") as f:
            json.dump([result.model_dump(mode="json") for result in results], f, indent=4, ensure_ascii=False)

    def process_generation_pipeline(self, lines_to_process = -1):
        results : list[PipelineLMResult] = []
        failed_lines : list[int] = []

        lines = self.read_input_lines(lines_to_process)

        self.logger.info("pipeline starts (run_id=%s, lines=%d, log=%s, partial=%s)",
                         self.logger.run_id, len(lines), self.logger.calls_path, self.partial_path)

        with open(self.partial_path, 'w', encoding="utf-8") as partial:
            for index, line in lines:
                with self.logger.context(trace_id=self.logger.new_trace(), line_idx=index):
                    self.logger.info("line %d: %s", index, line[:70])

                    try:
                        answer = self.generate_pipeline_step(UserMessage(content=line), step=index)
                    except Exception as exc:
                        self.logger.error("line %d crashed with %s: %s", index, type(exc).__name__, exc)
                        failed_lines.append(index)
                        continue

                    if answer is None:
                        failed_lines.append(index)
                        continue

                    results.append(answer)
                    self.append_partial(partial, answer)

        if results:
            self.write_output(results)
            self.logger.info("finished: %d/%d lines -> %s", len(results), len(lines), self.output_path)
        else:
            self.logger.error("pipeline ended without any results")

        if failed_lines:
            self.logger.warning("failed lines: %s", failed_lines)

        degraded_lines = [r.question_index for r in results if r.degraded_steps]

        if degraded_lines:
            self.logger.warning("degraded lines: %s", degraded_lines)

        summary = self.logger.summary()
        summary.update({
            "lines_total": len(lines),
            "lines_completed": len(results),
            "lines_failed": failed_lines,
            "lines_degraded": degraded_lines,
            "partial_log": self.partial_path,
        })

        self.logger.info("calls: %d, success: %d (%.0f%%), outcomes: %s",
                         summary["total_calls"], summary["ok_calls"],
                         summary["ok_rate"] * 100, summary["by_outcome"])
        self.logger.info("lines: %d/%d completed, %d degraded, %d failed",
                         len(results), len(lines), len(degraded_lines), len(failed_lines))

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