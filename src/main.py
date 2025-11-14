from config.llm_config import LlmConfig
from infrastructure.llms.gemini import GeminiLlm
from infrastructure.blackboards.json_blackboard import JsonBlackboard
from domain.agent import *
from application.orchestrator import Orchestrator

import os


llm_config = LlmConfig()

gemini_llm = GeminiLlm(llm_config, model_name='gemini-2.0-flash')

# Agents
solver_agent = SolverAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "solver_prompt.md"))
query_agent = QueryAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "query_prompt.md"))
domain_exp_agent = DomainExpertAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "domain_expert_prompt.md"))
coding_exp_agent = CodingExpertAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "coding_expert_prompt.md"))
writer_agent = WriterAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "writer_prompt.md"))

# Black boards
json_blackboard = JsonBlackboard(os.path.join(os.getcwd(), "attempts"))


orchestrator = Orchestrator(
    solver_agent,
    query_agent,
    coding_exp_agent,
    domain_exp_agent,
    writer_agent,
    json_blackboard
)

orchestrator.run("Design a full webpage for an ecommerce that has a main theme of Global warming. Decide what contents should be there")



