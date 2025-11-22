from config.llm_config import LlmConfig
from domain.interface.llm import LlmInterface
from pydantic import BaseModel
import requests
import json


class OpenRouterLlm(LlmInterface):
    def __init__(self, llm_config:LlmConfig, model_name:str="anthropic/claude-sonnet-4.5"):
        self.mode_name = model_name
        self.temperature = 0.5
        self.system_prompt = "You are a helpful assistant"