from __future__ import annotations
from typing import List, Tuple

from dataset import make_dataset
from search import bfs_proof_search
from state import ProofState


def evaluate_dataset(max_steps: int = 20) -> None:
    examples: List[Tuple[str, ProofState]] = make_dataset()

    solved = 0
    total = len(examples)

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for name, state in examples:
        result = bfs_proof_search(state, max_steps=max_steps)

        print(f"\nExample: {name}")
        print(state)
        print(f"Success: {result.success}")
        print(f"Visited states: {result.visited_states}")

        if result.success:
            solved += 1
            print("Tactic path:", " -> ".join(result.tactic_path))
        else:
            print("Tactic path: NONE")

    print("\n" + "=" * 60)
    print(f"Solved {solved}/{total}")
    print("=" * 60)