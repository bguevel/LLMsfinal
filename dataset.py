from __future__ import annotations
from typing import List, Tuple

from logic import And, Imp, Or, Var, Formula, is_tautology
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
        ("or_commute", theorem_to_state(Imp(Or(P, Q), Or(Q, P)))),
        ("and_intro_nested", theorem_to_state(Imp(P, Imp(Q, Imp(R, And(P, And(Q, R))))))),
    ]

    return examples


def make_truth_dataset() -> List[Tuple[str, Formula, bool]]:
    """
    Small validity dataset for the AST-embedding direction.

    The label means "is this propositional formula true under every assignment?"
    rather than "can the current tactic set prove it?".
    """
    P = Var("P")
    Q = Var("Q")
    R = Var("R")

    examples: List[Tuple[str, Formula, bool]] = [
        ("identity", Imp(P, P), True),
        ("const", Imp(P, Imp(Q, P)), True),
        ("and_intro", Imp(P, Imp(Q, And(P, Q))), True),
        ("or_commute", Imp(Or(P, Q), Or(Q, P)), True),
        ("not_every_or_is_true", Or(P, Q), False),
        ("bad_and_elim_reverse", Imp(P, And(P, Q)), False),
        ("affirm_consequent_shape", Imp(Imp(P, Q), Imp(Q, P)), False),
        ("nested_non_tautology", Imp(And(P, Or(Q, R)), Q), False),
    ]

    for name, formula, label in examples:
        actual = is_tautology(formula)
        if actual != label:
            raise AssertionError(f"Truth label mismatch for {name}: expected {label}, got {actual}")

    return examples
