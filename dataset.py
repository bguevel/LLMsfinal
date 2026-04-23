from __future__ import annotations
from typing import List, Tuple

from logic import And, Imp, Or, Var, Formula
from state import Goal, ProofState


def theorem_to_state(theorem: Formula) -> ProofState:
    return ProofState(goals=[Goal(assumptions=tuple(), target=theorem)])


def make_dataset() -> List[Tuple[str, ProofState]]:
    """
    A tiny collection of provable propositional formulas for testing.
    """
    P = Var("P")
    Q = Var("Q")
    R = Var("R")

    examples: List[Tuple[str, ProofState]] = [
        ("identity", theorem_to_state(Imp(P, P))),
        ("const", theorem_to_state(Imp(P, Imp(Q, P)))),
        ("and_intro", theorem_to_state(Imp(P, Imp(Q, And(P, Q))))),
        ("or_intro_left", theorem_to_state(Imp(P, Or(P, Q)))),
        ("or_intro_right", theorem_to_state(Imp(P, Or(Q, P)))),
        ("nested_identity", theorem_to_state(Imp(Imp(P, P), Imp(P, P)))),
        ("triple_const", theorem_to_state(Imp(P, Imp(Q, Imp(R, P))))),
    ]

    return examples