from __future__ import annotations

from dataset import make_truth_dataset
from statement_generation import check_statement_truth, load_labeled_statements


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
