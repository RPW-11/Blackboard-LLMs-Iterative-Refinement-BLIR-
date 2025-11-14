from config.llm_config import LlmConfig
from infrastructure.llms.gemini import GeminiLlm
from infrastructure.markdown_logger import print_markdown
from infrastructure.blackboards.json_blackboard import JsonBlackboard
from domain.solver_agent import SolverAgent

import os


llm_config = LlmConfig()

gemini_llm = GeminiLlm(llm_config, model_name='gemini-2.0-flash')
solver_agent = SolverAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "solver_prompt.md"))
res = solver_agent.process_message_stream("Write a code to list down job occupancies in tech using any APIs")

json_blackboard = JsonBlackboard(os.path.join(os.getcwd(), "attempts"))

code = ""

for text in res:
    print_markdown(text)
    code += text

json_blackboard.save_attempt({ "code": code })

print(json_blackboard.get_attempt_results())

