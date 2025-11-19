from .agent_base import AgentBase


class CodingExpertAgent(AgentBase):
    def __init__(self, llm, system_prompt_path):
        super().__init__(llm, system_prompt_path)


    def process_message(self, message) -> str:
        return self.llm.invoke(message)


    def process_message_stream(self, message):
        pass
    
