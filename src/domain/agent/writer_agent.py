from .agent_base import AgentBase

class WriterAgent(AgentBase):
    def __init__(self, llm, system_prompt_path):
        super().__init__(llm, system_prompt_path)


    def process_message(self, message) -> str:
        return f"Answering: {message}\nWriter agent feedback"
    

    def process_feedback(self, domain_feedback: str, coding_feedback: str) -> str:
        message = f"Domain feedback:\n{domain_feedback}\n\nCoding feedback:\n{coding_feedback}"
        return self.process_message(message)


    def process_message_stream(self, message):
        pass