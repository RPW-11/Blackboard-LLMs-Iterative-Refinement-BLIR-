from pydantic import BaseModel, Field


class QueryAgentResponse(BaseModel):
    domain_expert_prompt: str = Field(description="The prompt that will be given to the Domain Expert agent")
    coding_expert_prompt: str = Field(description="The prompt that will be given to the Coding Expert agent")


