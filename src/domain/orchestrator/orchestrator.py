from domain.agent import *
from domain.interface.blackboard import BlackboardInterface
from infrastructure.markdown_logger import print_markdown
from pydantic import BaseModel
from typing import List


class Orchestrator:
    def __init__(
        self,
        solver_agent:SolverAgent,
        query_agent: QueryAgent,
        coding_expert_agent: CodingExpertAgent,
        domain_expert_agent: DomainExpertAgent,
        writer_agent: WriterAgent,
        black_board: BlackboardInterface
    ):
        self.solver_agent = solver_agent
        self.query_agent = query_agent
        self.coding_expert_agent = coding_expert_agent
        self.domain_expert_agent = domain_expert_agent
        self.writer_agent = writer_agent
        self.black_board = black_board
        self.solver_prompt = ""


    def run(self, problem_definition: str, response_structure: BaseModel) -> dict | None:
        try:
            distRes = self.solver_agent.process_message_structured(problem_definition, response_structure)
            return distRes
        
        except Exception as e:
            print(f"Error from LLM: {e}")
            return None

    def save_to_blackboard(self, data: dict):
        self.black_board.save_attempt(data)

    def load_results(self, n: int) -> List[dict]:
        return self.black_board.get_attempt_results(n)






        

    
        