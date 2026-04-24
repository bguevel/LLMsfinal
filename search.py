from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List, Set

from state import ProofState, serialize_state
from tactics import TACTIC_ORDER, apply_tactic


@dataclass
class SearchStep:
    depth: int
    state_before: str
    llm_ranked_tactics: List[str]
    tried_tactics: List[str]
    valid_tactics_taken: List[str]


@dataclass
class SearchResult:
    success: bool
    tactic_path: List[str]
    visited_states: int
    final_state: ProofState | None
    trace: List[SearchStep] = field(default_factory=list)


def bfs_proof_search(initial_state: ProofState, max_steps: int = 20) -> SearchResult:
    queue = deque([(initial_state, [])])
    visited: Set[str] = set()
    visited_states = 0

    while queue:
        state, path = queue.popleft()
        visited_states += 1

        if state.is_solved():
            return SearchResult(True, path, visited_states, state)

        key = serialize_state(state)
        if key in visited:
            continue
        visited.add(key)

        if len(path) >= max_steps:
            continue

        for tactic in TACTIC_ORDER:
            nxt = apply_tactic(state, tactic)
            if nxt is not None:
                queue.append((nxt, path + [tactic]))

    return SearchResult(False, [], visited_states, None)


def llm_topk_bfs(
    initial_state: ProofState,
    tactic_model,
    top_k: int = 2,
    max_steps: int = 20,
) -> SearchResult:
    """
    LLM-guided BFS.

    Important:
    This search only tries the LLM's top-k tactics.
    If the correct tactic is outside top-k, the search can fail.
    """
    queue = deque([(initial_state, [])])
    visited: Set[str] = set()
    visited_states = 0
    trace: List[SearchStep] = []

    while queue:
        state, path = queue.popleft()
        visited_states += 1

        if state.is_solved():
            return SearchResult(True, path, visited_states, state, trace)

        key = serialize_state(state)
        if key in visited:
            continue
        visited.add(key)

        if len(path) >= max_steps:
            continue

        llm_order = tactic_model.predict_tactic_order(state)
        chosen_tactics = llm_order[:top_k]

        tried = []
        valid_taken = []

        for tactic in chosen_tactics:
            tried.append(tactic)
            nxt = apply_tactic(state, tactic)

            if nxt is not None:
                valid_taken.append(tactic)
                queue.append((nxt, path + [tactic]))

        trace.append(
            SearchStep(
                depth=len(path),
                state_before=str(state),
                llm_ranked_tactics=llm_order,
                tried_tactics=tried,
                valid_tactics_taken=valid_taken,
            )
        )

    return SearchResult(False, [], visited_states, None, trace)