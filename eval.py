from __future__ import annotations
from typing import List, Tuple

from dataset import make_dataset, make_truth_dataset
from search import bfs_proof_search
from state import ProofState
from statement_generation import check_statement_truth, load_labeled_statements


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


def evaluate_truth_dataset() -> None:
    examples = make_truth_dataset()

    print("=" * 60)
    print("TRUTH EVALUATION DATASET")
    print("=" * 60)

    for name, formula, label in examples:
        print(f"\nExample: {name}")
        print(formula)
        print(f"Tautology label: {label}")


def evaluate_saved_statement_file(input_path: str) -> None:
    examples = load_labeled_statements(input_path)

    print("=" * 60)
    print("SAVED TRUTH EVALUATION FILE")
    print("=" * 60)
    print(f"File: {input_path}")

    correct = 0
    for statement in examples:
        checked = check_statement_truth(statement.formula)
        correct += int(checked == statement.label)

        print(f"\nExample: {statement.name}")
        print(statement.formula)
        print(f"Stored label: {statement.label}")
        print(f"Checked label: {checked}")

    print("\n" + "=" * 60)
    print(f"Labels verified {correct}/{len(examples)}")
    print("=" * 60)
