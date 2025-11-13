from config.llm_config import LlmConfig
from domain.interface.llm import LlmInterface
from google import genai
from google.genai import types
from pydantic import BaseModel


class GeminiLlm(LlmInterface):
    def __init__(self, llm_config:LlmConfig, model_name:str="gemini-2.5-flash"):
        self.mode_name = model_name
        self.temperature = 0.5
        self.system_prompt = "You are a helpful assistant"
        self.client = genai.Client(api_key=llm_config.GEMINI_API_KEY)
        
    
    def invoke(self, prompt: str, structure_schema:BaseModel=None):
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json" if structure_schema else None,
                response_json_schema=structure_schema.model_json_schema() if structure_schema else None
            )

            response = self.client.models.generate_content(
                model=self.mode_name,
                config=config,
                contents=prompt
            )

            if structure_schema:
                return structure_schema.model_validate_json(response.text)
            
            return response.text
        except Exception as e:
            raise Exception(f"Error occured while using gemini: {str(e)}")
        

    def invoke_stream(self, prompt: str):
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
            )

            response = self.client.models.generate_content_stream(
                model=self.mode_name,
                config=config,
                contents=prompt
            )
            
            for chunk in response:
                yield chunk.candidates[0].content.parts[0].text
            
        except Exception as e:
            raise Exception(f"Error occured while using gemini: {str(e)}")

    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    
    def set_temperature(self, temperature):
        self.temperature = temperature
    
