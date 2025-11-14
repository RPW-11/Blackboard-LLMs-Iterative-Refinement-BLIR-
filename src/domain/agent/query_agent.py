from .agent_base import AgentBase
from ..response import QueryAgentResponse


class QueryAgent(AgentBase):
    def __init__(self, llm, system_prompt_path):
        super().__init__(llm, system_prompt_path)


    def process_message(self, message)-> QueryAgentResponse:
        return self.llm.invoke(prompt=message, structure_schema=QueryAgentResponse)


    def process_message_stream(self, message):
        """No implementation for this function because it will output a structured output"""
        pass    