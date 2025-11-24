from config.llm_config import LlmConfig
from domain.interface.llm import LlmInterface
from pydantic import BaseModel
from ..markdown_logger import print_markdown
import requests
import json


class OpenRouterLlm(LlmInterface):
    def __init__(self, llm_config:LlmConfig, model_name:str="deepseek/deepseek-chat-v3-0324:free"):
        self.mode_name = model_name
        self.temperature = 0.5
        self.system_prompt = "You are a helpful assistant"
        self.header = {
            "Authorization": f"Bearer {llm_config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
    
    def invoke(self, prompt: str, structure_schema:BaseModel=None) -> BaseModel | str:
        data = {
            "model": self.mode_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "temperature": self.temperature
                }
            ],
            # 'provider': {
            #     'require_parameters': True,
            # },
        }

        if structure_schema:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": structure_schema.model_json_schema()
                }
            }

        data = json.dumps(data, indent=4)

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=self.header,
            data=data
        )
        response = response.json()
        print(response)
        response = response['choices'][0]['message']["content"]

        if structure_schema:
            print_markdown(response)
            return structure_schema.model_validate_json(response)

        print_markdown(response)
        return response
    

    def invoke_stream(self, prompt: str):
        raise NotImplementedError("Not implemented yet")
    

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt


    def set_temperature(self, temperature: int):
        self.temperature = temperature