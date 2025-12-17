from config.llm_config import LlmConfig
from infrastructure.llms.gemini import GeminiLlm
from infrastructure.llms.openrouter import OpenRouterLlm
from infrastructure.blackboards.json_blackboard import JsonBlackboard
from infrastructure.util import parse_solomon_instance, read_md
from infrastructure.markdown_logger import MarkdownTerminalRenderer
from domain.agent import *
from domain.problem.cvrp_ga import CVRPGeneticAlgorithm
from domain.orchestrator.orchestrator import Orchestrator

import os


llm_config = LlmConfig()

gemini_llm = GeminiLlm(llm_config, model_name='gemini-2.5-flash')
qwen_llm = OpenRouterLlm(llm_config)

# Agents
solver_agent = SolverAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "solver_prompt.md"))
query_agent = QueryAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "query_prompt.md"))
domain_exp_agent = DomainExpertAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "domain_expert_prompt.md"))
coding_exp_agent = CodingExpertAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "coding_expert_prompt.md"))
writer_agent = WriterAgent(gemini_llm, os.path.join(os.getcwd(), "prompts", "writer_prompt.md"))

# Parser Solomon
datasets = os.listdir("problems/solomon/dataset")

for file in datasets[:2]:
    filename = file.split(".")[0]

    if (os.path.exists(os.path.join(os.getcwd(), "attempts", filename))):
        continue

    # Black boards
    json_blackboard = JsonBlackboard(os.path.join(os.getcwd(), "attempts", filename))

    # Orchestrator
    orchestrator = Orchestrator(
        solver_agent,
        query_agent,
        coding_exp_agent,
        domain_exp_agent,
        writer_agent,
        json_blackboard
    )


    instance = parse_solomon_instance(os.path.join("problems/solomon/dataset", datasets[1]))
    description = read_md("problems/solomon/description.md")

    # Logger
    logger = MarkdownTerminalRenderer()

    ga = CVRPGeneticAlgorithm(description, orchestrator, logger, instance, population_size=100)

    result = ga.solve()

    for route in result.routes:
        print("Route:", route)

    print("Total Distance:", result.total_distance)
    print("Total Vehicles:", result.num_vehicles)


