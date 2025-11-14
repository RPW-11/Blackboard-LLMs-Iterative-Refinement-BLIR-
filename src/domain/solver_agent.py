from .agent_base import AgentBase


class SolverAgent(AgentBase):
    def __init__(self, llm, system_prompt_path):
        super().__init__(llm, system_prompt_path)


    def process_message(self, message):
        return self.llm.invoke(prompt=message)


    def process_message_stream(self, message):
        return self.llm.invoke_stream(message)    