from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from state import ProofState, serialize_state
from tactics import TACTIC_ORDER, apply_tactic


@dataclass
class SearchResult:
    success: bool
    tactic_path: List[str]
    visited_states: int
    final_state: ProofState | None

def llm_guided_search(initial_state: ProofState, tactic_model, max_steps: int = 20) -> SearchResult:
    """
    Greedy search guided by the LLM's top tactic proposal.
    Falls back to trying all tactics if the proposed one is invalid.
    """
    state = initial_state
    path: List[str] = []
    visited: Set[str] = set()
    visited_states = 0

    for _ in range(max_steps):
        visited_states += 1

        if state.is_solved():
            return SearchResult(True, path, visited_states, state)

        key = serialize_state(state)
        if key in visited:
            break
        visited.add(key)

        proposed = tactic_model.predict_tactic(state)

        # Try model proposal first
        if proposed in TACTIC_ORDER:
            nxt = apply_tactic(state, proposed)
            if nxt is not None:
                state = nxt
                path.append(proposed)
                continue

        # Fallback: brute-force valid tactic if proposal failed
        applied = False
        for tactic in TACTIC_ORDER:
            nxt = apply_tactic(state, tactic)
            if nxt is not None:
                state = nxt
                path.append(tactic)
                applied = True
                break

        if not applied:
            break

    return SearchResult(False, path, visited_states, state)

def bfs_proof_search(initial_state: ProofState, max_steps: int = 20) -> SearchResult:
    """
    Breadth-first search over tactic sequences.
    """
    queue = deque([(initial_state, [])])
    visited: Set[str] = set()
    visited_states = 0

    while queue:
        state, path = queue.popleft()
        visited_states += 1

        if state.is_solved():
            return SearchResult(
                success=True,
                tactic_path=path,
                visited_states=visited_states,
                final_state=state,
            )

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

    return SearchResult(
        success=False,
        tactic_path=[],
        visited_states=visited_states,
        final_state=None,
    )