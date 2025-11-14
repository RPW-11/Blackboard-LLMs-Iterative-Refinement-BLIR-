from domain.agent import *
from domain.interface.blackboard import BlackboardInterface
from infrastructure.markdown_logger import print_markdown


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


    def run(self, problem_definition: str):
        res = self.solver_agent.process_message_stream(
            f"Problem:\n{problem_definition}\n\nInstruction:\n{self.solver_prompt}\n\n" if self.solver_prompt != "" else  f"Problem:\n{problem_definition}\n\n" 
        )

        code = ""
        for text in res:
            print_markdown(text)
            code += text

        self.black_board.save_attempt({ "code": code })

        print("="*20 + "Query agent thinking..." + "="*20)
        query_prompt = self.query_agent.process_message(f"Probelm:\n{problem_definition}\n\nSolver code: {code}")

        print(f"Domain expert prompt:\n{query_prompt.domain_expert_prompt}\n\n")
        print(f"Coding expert prompt:\n{query_prompt.coding_expert_prompt}")

        # should be async processing
        domain_feedback = self.domain_expert_agent.process_message(query_prompt.domain_expert_prompt)
        coding_feedback = self.coding_expert_agent.process_message(query_prompt.coding_expert_prompt)

        self.solver_prompt = self.writer_agent.process_feedback(domain_feedback, coding_feedback)




        

    
        