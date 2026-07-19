import os
from dotenv import load_dotenv
from promts import *
from llm_api import *
from data_models import *
import json

MAX_RETRIES = 3

class MultiStepLLMPipeline:
    def __init__(self, extract_client : ExtractMeaningLM,
                 classifier_client : IntentClassifierLM,
                 field_extractor_client : FieldExtractorLM,
                 response_generator_client : ResponseGeneratorLM,
                 judge_client : SelfCheckLM,
                 input_path : str,
                 output_path : str):
        self.extractClient = extract_client
        self.classifierClient = classifier_client
        self.fieldExtractorClient = field_extractor_client
        self.responseGeneratorClient = response_generator_client
        self.judgeClient = judge_client
        self.input_path = input_path
        self.output_path = output_path
    
    def extract_sense(self, msg : UserMessage):
        print("Start sense extraction step")
        
        for i in range(MAX_RETRIES):
            extracted_sense : str = self.extractClient.generate([msg])
            
            if extracted_sense is not None:
                break
        else:
            return None
        
        print(f"Extracted data : {extracted_sense}\n")
        
        return extracted_sense
    
    def generate_intent(self, msg : UserMessage):
        print("Start intent classifier step")
        
        for i in range(MAX_RETRIES):
            intent_answer = self.classifierClient.generate_classification(msg)
            if IntentLMAnswer.validate_answer_json(llm_answer=intent_answer):
                break
        else:
            return None
        
        result = IntentLMAnswer(**json.loads(intent_answer))

        print(f"Intent classified as {intent_answer}\n")

        return result.intent.value
    
    def extract_fields(self, msg : UserMessage, intent):
        print("Start fields extraction step")
        
        model_type = INTENT_MODELS[intent]
        
        self.fieldExtractorClient.sys_message = SystemMessage(content=COMMON_EXTRACTION_PROMPT + "\n" + EXTRACTION_PROMPT_REGISTRY[intent])
        self.fieldExtractorClient.set_response_schema_format(construct_api_payload(model_type.model_json_schema()))
                
        for i in range(MAX_RETRIES):
            answer = self.fieldExtractorClient.generate([msg])
            
            try:
                data = json.loads(answer)
                validated_answer = model_type(**data)
                break
            except Exception as ex:
                print(f"Field extraction has been failed with : {ex}, answer : {answer}")
        else:
            print("Field extraction has been failed")
            return None
        
        print(f"Fields extracted as {answer}\n")
        
        return validated_answer
    
    def generate_final_answer(self, msg : UserMessage, sense : str, intent : str, sentiment : str, field_extraction : str):
        print("Start final answer generation step")
        
        self.responseGeneratorClient.sys_message = SystemMessage(content=get_final_asnwer_prompt())
        for i in range(MAX_RETRIES):
            message = UserMessage(content=build_final_answer_message(msg.content,
                                                                     intent=intent,
                                                                     sentiment=sentiment,
                                                                     sense=sense,
                                                                     field_extraction=field_extraction))
            answer = self.responseGeneratorClient.generate([message])
            if answer is not None:
                print(f"Final answer generated as: {answer}\n")
                return answer
        else:
            return None    
    
    def generate_judge_results(self, start_msg : str, final_answer : str):
        print("Start judge step")
        
        def validate(judge_answer):
            try:
                data = json.loads(judge_answer)
                answer = JudgedLMResult(**data)
                return True
            except Exception as ex:
                return False
            
        for i in range(MAX_RETRIES):
            result = self.judgeClient.generate([UserMessage(content=get_judge_prompt(start_msg, final_answer))])
            if validate(result):
                break
            else:
                print("Judge result validation has been failed")
        else:
            return None
        
        print(f"Judge step finished with result: {result}")
        
        return result
            
    def generate_pipeline_step(self, msg : UserMessage, step : int = -1) -> PipelineLMResult | None:
        print(f"\nStart generation step {step}")
        
        sense = self.extract_sense(msg=msg)
        
        if sense is None:
            print("There is an error with sense generation")
            return None            
        
        intent = self.generate_intent(msg=msg)
        
        if intent is None:
            print("There is an error with intent classification")
            return None            
        
        field_extraction = self.extract_fields(msg=msg, intent=intent)
        
        if field_extraction is None:
            print("There is an error with fields extraction")
            return None             
        
        final_answer = self.generate_final_answer(msg, sense, intent, field_extraction.sentiment.value if field_extraction.sentiment is not None else "None", field_extraction)
        
        if final_answer is None:
            print("There is an error with final answer generation")
            return None
        
        judge_result = self.generate_judge_results(msg.content, final_answer)
        
        if judge_result is None:
            print("There is an error with judge result generation")
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
        
        print("Start pipeline generation\n")
        
        with open(self.input_path, 'r') as f:
            for i, line in enumerate(f):
                
                if lines_gate and (i >= lines_to_process):
                    break
                
                line = line.strip()
                answer = self.generate_pipeline_step(UserMessage(content=line), step=i)
                last_index  = i
                
                if answer is not None:
                    results.append(answer)
        
        if len(results) == 0:
            print("Pipeline generation failed")
            return
                                
        with open(self.output_path, 'w') as f:
            json.dump([result.model_dump(mode="json") for result in results], f, indent=4, ensure_ascii=False)
            
        print("\nPipeline generation success")
        
        processed_indexes = {result.question_index for result in results}
        problematic_points = [i for i in range(last_index + 1) if i not in processed_indexes]
        
        if len(problematic_points) > 0:
            print(f"Lines, that have not been processed : {problematic_points}")

@dataclass
class LLMPipelineSettings:
    general_model_name : str
    general_model_url : str
    general_model_api_key : str
    simple_model_name : str
    simple_model_url : str
    simple_model_api_key : str

def build_llm_pipeline(settings : LLMPipelineSettings, input_path : str, output_path : str) -> MultiStepLLMPipeline: 
    
    print("Start building LLM pipeline") # i could use just base LLMClient class from llm_api.py, but i thought all those LM classes would have some different behaviours
    
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
                                 payload_schema=construct_api_payload(JudgedLMResult.model_json_schema()), # i actually could use it before for other cases, but i learned about it not far time ago i realized this feature
                                 sys_message=SystemMessage(content=SELF_CHECK_PROMPT),
                                 temperature=.01)
    
    judge_client = SelfCheckLM(api_key=settings.general_model_api_key,
                               llm_settings=judge_settings)
    
    print("LLM pipeline build sucess")
    
    return MultiStepLLMPipeline(extract_client=meaning_client,
                                classifier_client=intent_client,
                                field_extractor_client=field_extractor_client,
                                response_generator_client=response_generator_client,
                                judge_client=judge_client,
                                input_path=input_path,
                                output_path=output_path)