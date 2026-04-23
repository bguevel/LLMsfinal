from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from logic import Formula, pretty_formula


@dataclass(frozen=True)
class Goal:
    assumptions: Tuple[Formula, ...]
    target: Formula

    def __str__(self) -> str:
        if self.assumptions:
            asm = ", ".join(pretty_formula(a) for a in self.assumptions)
        else:
            asm = "∅"
        return f"{asm} ⊢ {pretty_formula(self.target)}"


@dataclass
class ProofState:
    goals: List[Goal] = field(default_factory=list)

    def is_solved(self) -> bool:
        return len(self.goals) == 0

    def first_goal(self) -> Goal | None:
        return self.goals[0] if self.goals else None

    def copy(self) -> "ProofState":
        return ProofState(goals=list(self.goals))

    def __str__(self) -> str:
        if self.is_solved():
            return "SOLVED"
        lines = ["ProofState:"]
        for i, g in enumerate(self.goals, start=1):
            lines.append(f"  {i}. {g}")
        return "\n".join(lines)


def serialize_state(state: ProofState) -> str:
    return repr(state.goals)