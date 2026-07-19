import argparse
import os
from dotenv import load_dotenv
from promts import *
from llm_api import *
from data_models import *
import json
from multistep_llm_pipeline import (
    MultiStepLLMPipeline,
    LLMPipelineSettings,
    build_llm_pipeline,
    add_llm_arguments,
    settings_from_args,
)

env = load_dotenv()
main_api_key = os.getenv("DEEP_INFRA_KEY")
classification_api_key = os.getenv("OPEN_AI_KEY")

intent_model = "gpt-4o-mini"
model_name = "deepseek-ai/DeepSeek-V4-Flash"
deepinfa_url = "https://api.deepinfra.com/v1/openai"

temperature = .85
top_p = .9
max_tokens = 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

intent_input_path = os.path.join(BASE_DIR, "intent/questions.txt")
intent_classification_output = os.path.join(BASE_DIR, "intent/classification.txt")

messages_output_path = os.path.join(BASE_DIR, "llm_datasets/")
output_file = "results"
output_format = ".txt"
messages_input_path = os.path.join(BASE_DIR, "llm_datasets/inputs.txt")

llm_pipeline_input = os.path.join(BASE_DIR, "llm_pipeline_results/input.txt")
llm_pipeline_output = os.path.join(BASE_DIR, "llm_pipeline_results/output.txt")

schemas = {LLMSummarizationAnswer : (PAYLOAD_SCHEMA, LLMSummarizationAnswer.validate_answer_json),
           LLMSummarizationAnswer_LLM_DAY3 : (PAYLOAD_SCHEMA_LLM_DAY_3, LLMSummarizationAnswer_LLM_DAY3.validate_answer_json)}

def process_messages(path : str, client : LLMClient, path_suffix : str = "", max_retries = 3):
    results = []
            
    with open(path, 'r') as f:
        for line in f:
            
            for i in range(max_retries): 
                answer = client.generate([UserMessage(content=line)])
                if schemas[client.payload_type][1](llm_answer=answer):
                    results.append(answer)
                    break
            else:
                print(f"Failed to process: {line}")

    output = messages_output_path + output_file + "_" + path_suffix + output_format

    if len(results) < 1:
        print("Failed to process all messages")
        return
    
    with open(output, 'w') as f:
        json.dump([json.loads(x) for x in results], f, indent=4)

def iterate_registry_settings(client : LLMClient, settings : LLMSettings):
    for name, prompt in PROMPT_REGISTRY.items():
        settings.sys_message = SystemMessage(content=prompt)
        client.apply_settings(settings)
        process_messages(messages_input_path, client, name)

def process_messages_with_intent_classification(llmClient : LLMClient, classifier : IntentClassifierLM, settings : LLMSettings):
    MAX_RETRIES = 3
    intent_results = []
    results = []
    
    settings.payload_schema = LLM_WITH_INTENT_ANSWER_SCHEMA
    
    with open(intent_input_path, "r") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            for i in range(MAX_RETRIES):
                intent_answer = classifier.generate_classification(UserMessage(content=line))
                if IntentLMAnswer.validate_answer_json(llm_answer=intent_answer):
                    print(intent_answer)
                    intent_results.append(intent_answer)
                    break
            else:
                print(f"Failed to process: {line}")
                continue
            
            result = IntentLMAnswer(**json.loads(intent_answer))
            settings.sys_message = SystemMessage(content=PROMPT_REGISTRY_DAY_4[result.intent.value])
            llmClient.apply_settings(settings)
            
            answer = llmClient.generate([UserMessage(content=line)])
            print("\n" + answer + "\n")
            results.append(answer)
            
    with open(intent_classification_output, "w") as out:
        json.dump([json.loads(answer) for answer in intent_results], out, indent=4)
        
    with open(messages_output_path + output_file + "_intent_answers" + output_format, 'w') as f:
        f.write("".join((f"{i}: " + result + '\n' for i, result in enumerate(results))))
        
def parse_args():
    parser = argparse.ArgumentParser(
        prog="main",
        description="Run the multi-step LLM pipeline over a file of input messages.")
    add_llm_arguments(parser)
    parser.add_argument("--lines", type=int, default=10,
                        help="how many lines to read from the input file (-1 for all)")
    parser.add_argument("--input", default=llm_pipeline_input, help="input file with one message per line")
    parser.add_argument("--output", default=llm_pipeline_output, help="where to write the JSON results")
    return parser.parse_args()

def main():
    args = parse_args()

    pipeline_settings = settings_from_args(args)

    multistepPipeline = build_llm_pipeline(pipeline_settings, input_path=args.input, output_path=args.output)
    multistepPipeline.process_generation_pipeline(lines_to_process=args.lines)

if __name__ == "__main__":
    main()


"""
payload_type : type = LLMSummarizationAnswer_LLM_DAY3

settings = LLMSettings(model_name=model_name,
                       base_url=deepinfa_url,
                       payload_type=payload_type,
                       payload_schema=schemas[payload_type][0],
                       sys_message= SystemMessage(content=next(iter(PROMPT_REGISTRY.values()), None)),
                       max_tokens=max_tokens,
                       temperature=temperature,
                       top_p=top_p)

classifier = IntentClassifierLM(classification_api_key, 
                                intent_model,
                                SystemMessage(content=INTENT_PROMPT),
                                INTENT_PAYLOAD_SCHEMA)

client = LLMClient(api_key=main_api_key, llm_settings=settings)




process_messages_with_intent_classification(llmClient=client, classifier=classifier, settings=settings)
process_messages(messages_input, client, "llm_day_3")
iterate_registry_settings(client, settings=settings)
"""