from config.llm_config import LlmConfig
from infrastructure.llms.gemini import GeminiLlm
from domain.solver_agent import SolverAgent

import os


llm_config = LlmConfig()

gemini_llm = GeminiLlm(llm_config)
solver_agent = SolverAgent(gemini_llm, os.path.join(os.getcwd(), "prompts/solver_prompt.md"))
res = solver_agent.process_message_stream("Talk to me about Plato philoshophy")

for text in res:
    print(text)