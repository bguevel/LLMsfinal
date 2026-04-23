from __future__ import annotations
from typing import Callable, Dict, List, Optional

from logic import And, Imp, Or
from state import Goal, ProofState


TacticFn = Callable[[ProofState], Optional[ProofState]]

def cases(state: ProofState) -> Optional[ProofState]:
    """
    OR elimination:
    If an assumption is A \/ B, split into two cases:
      - assume A, prove target
      - assume B, prove target
    """
    if state.is_solved():
        return None

    g = state.first_goal()
    assert g is not None

    for i, asm in enumerate(g.assumptions):
        if isinstance(asm, Or):
            new_state = state.copy()

            # remove the A \/ B assumption
            remaining = g.assumptions[:i] + g.assumptions[i+1:]

            # create two subgoals
            goal_left = Goal(
                assumptions=remaining + (asm.left,),
                target=g.target
            )
            goal_right = Goal(
                assumptions=remaining + (asm.right,),
                target=g.target
            )

            # replace current goal with two new ones
            rest = new_state.goals[1:]
            new_state.goals = [goal_left, goal_right, *rest]

            return new_state

    return None

def assumption(state: ProofState) -> Optional[ProofState]:
    """
    If target is already one of the assumptions, solve the current goal.
    """
    if state.is_solved():
        return None

    g = state.first_goal()
    assert g is not None

    if g.target in g.assumptions:
        new_state = state.copy()
        new_state.goals.pop(0)
        return new_state
    return None


def intro(state: ProofState) -> Optional[ProofState]:
    """
    If target is A -> B, move A into assumptions and replace target with B.
    """
    if state.is_solved():
        return None

    g = state.first_goal()
    assert g is not None

    if isinstance(g.target, Imp):
        new_state = state.copy()
        new_state.goals[0] = Goal(
            assumptions=g.assumptions + (g.target.left,),
            target=g.target.right,
        )
        return new_state
    return None


def split(state: ProofState) -> Optional[ProofState]:
    """
    If target is A /\ B, create two subgoals: prove A and prove B.
    """
    if state.is_solved():
        return None

    g = state.first_goal()
    assert g is not None

    if isinstance(g.target, And):
        new_state = state.copy()
        rest = new_state.goals[1:]
        new_state.goals = [
            Goal(g.assumptions, g.target.left),
            Goal(g.assumptions, g.target.right),
            *rest,
        ]
        return new_state
    return None


def left(state: ProofState) -> Optional[ProofState]:
    """
    If target is A \/ B, try proving A.
    """
    if state.is_solved():
        return None

    g = state.first_goal()
    assert g is not None

    if isinstance(g.target, Or):
        new_state = state.copy()
        new_state.goals[0] = Goal(g.assumptions, g.target.left)
        return new_state
    return None


def right(state: ProofState) -> Optional[ProofState]:
    """
    If target is A \/ B, try proving B.
    """
    if state.is_solved():
        return None

    g = state.first_goal()
    assert g is not None

    if isinstance(g.target, Or):
        new_state = state.copy()
        new_state.goals[0] = Goal(g.assumptions, g.target.right)
        return new_state
    return None


TACTIC_FUNCS = {
    "assumption": assumption,
    "intro": intro,
    "split": split,
    "left": left,
    "right": right,
    "cases": cases,   
}

TACTIC_ORDER = [
    "assumption",
    "intro",
    "split",
    "left",
    "right",
    "cases",   
]


def applicable_tactics(state: ProofState) -> List[str]:
    """
    Return all tactics that can be legally applied to the current state.
    """
    valid = []
    for name in TACTIC_ORDER:
        nxt = TACTIC_FUNCS[name](state)
        if nxt is not None:
            valid.append(name)
    return valid


def apply_tactic(state: ProofState, tactic_name: str) -> Optional[ProofState]:
    fn = TACTIC_FUNCS.get(tactic_name)
    if fn is None:
        raise ValueError(f"Unknown tactic: {tactic_name}")
    return fn(state)