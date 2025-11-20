from .agent_base import AgentBase
from pydantic import BaseModel


class SolverAgent(AgentBase):
    def __init__(self, llm, system_prompt_path):
        super().__init__(llm, system_prompt_path)


    def process_message_structured(self, message: str, structured_output: BaseModel) -> dict:
        result = self.llm.invoke(message, structured_output)
        return result.model_dump()


    def process_message(self, message):
        print("="*30 + "using this prompt" + "="*30)
        print(self.system_prompt)
        return self.llm.invoke(prompt=message)


    def process_message_stream(self, message):
        print("="*30 + "using this prompt" + "="*30)
        print(self.system_prompt)
        return self.llm.invoke_stream(message)    