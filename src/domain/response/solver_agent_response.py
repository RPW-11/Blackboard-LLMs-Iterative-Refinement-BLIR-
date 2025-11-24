from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class OperatorCode(BaseModel):
    code: str = Field(..., description="Complete Python function definition as a string. Must be executable with exec(). No need to import libraries since all necessary libraries have been imported.")
    prob: float = Field(..., ge=0.0, le=1.0, description="Probability weight for selecting this operator.")
    name: Optional[str] = Field(None, description="Optional human-readable name, e.g. 'cluster_preserving_cx'")

    @field_validator("code")
    @classmethod
    def must_be_def(cls, v: str) -> str:
        stripped = v.strip()
        has_def = any(line.strip().startswith("def") for line in stripped.splitlines())
        if not (has_def and "(" in stripped and ")" in stripped):
            raise ValueError("Code must be a complete 'def function_name(...):' block")
        return v

class ScheduleConfig(BaseModel):
    interval: int = Field(..., ge=5, description="New large agent invocation interval in generations")
    early_stopping_generations_without_improvement: Optional[int] = None
    trigger_on_feasibility_drop_below: Optional[float] = Field(None, ge=0.0, le=1.0)


class LargeAgentResponse(BaseModel):
    crossover: List[OperatorCode] = Field(
        default_factory=list,
        description="New or updated crossover operators. Empty = keep current ones. If there is only one operator, you must write another one"
    )

    mutation: List[OperatorCode] = Field(
        default_factory=list,
        description="New or updated mutation operators. Empty = keep current ones."
    )

    repair: Optional[str] = Field(
        None,
        description="Full 'def repair(chromosome: List[int]) -> List[int]: ...' function as string. None = no change."
    )

    local_search: Optional[str] = Field(
        None,
        description="Full 'def local_search(chromosome: List[int]) -> List[int]: ...' function. None = no change."
    )

    schedule: Optional[ScheduleConfig] = Field(None, description="New scheduling policy")

    initial_population_prompt: Optional[str] = Field(
        None,
        description="If provided, replace the small LLM's initial population prompt for future runs on similar instances."
    )

    meta_notes: Optional[str] = Field(None, description="Any observations for long-term memory (e.g. 'on C-type instances with load_factor > 0.95, always start with split-then-route')")

    @field_validator("crossover", "mutation", mode="before")
    @classmethod
    def normalize_to_list(cls, v):
        return v or []

    def has_any_operator_changes(self) -> bool:
        return bool(self.crossover or self.mutation or self.repair or self.local_search)