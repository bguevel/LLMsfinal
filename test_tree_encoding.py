from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from dataset import make_dataset, make_truth_dataset
from statement_generation import (
    check_statement_truth,
    generate_and_save_labeled_statement_counts,
    generate_and_save_labeled_statements,
    load_labeled_statements,
    statement_text_for_formula,
)
from tree_encoding import (
    HashingTreeEmbedder,
    formula_to_path_entries,
    formula_to_prefix_tokens,
    formula_to_sexpr,
    path_entries_to_formula,
    prefix_tokens_to_formula,
    sexpr_to_formula,
)


def test_formula_codecs_round_trip() -> None:
    for _, state in make_dataset():
        for goal in state.goals:
            formulas = [*goal.assumptions, goal.target]
            for formula in formulas:
                assert prefix_tokens_to_formula(formula_to_prefix_tokens(formula)) == formula
                assert sexpr_to_formula(formula_to_sexpr(formula)) == formula
                assert path_entries_to_formula(formula_to_path_entries(formula)) == formula


def test_truth_dataset_labels_are_checked() -> None:
    assert make_truth_dataset()


def test_hash_embedding_shape_and_stability() -> None:
    if torch is None:
        print("skipping hash embedding check because torch is not installed")
        return

    embedder = HashingTreeEmbedder(dim=64)
    _, state = make_dataset()[0]

    first = embedder.encode_state(state)
    second = embedder.encode_state(state)

    assert first.shape == (64,)
    assert torch.equal(first, second)
    assert torch.linalg.vector_norm(first).item() > 0


def test_generated_statement_file_round_trip() -> None:
    path = Path("_generated_statement_test.jsonl")
    try:
        statements = generate_and_save_labeled_statements(
            n=12,
            output_path=path,
            max_depth=2,
            seed=7,
        )
        loaded = load_labeled_statements(path)

        assert len(statements) == 12
        assert len(loaded) == 12
        assert any(statement.label for statement in loaded)
        assert any(not statement.label for statement in loaded)
        assert all(check_statement_truth(statement.formula) == statement.label for statement in loaded)
        assert all(statement.text.startswith("is this statement true ") for statement in loaded)
        assert all(statement.text == statement_text_for_formula(statement.formula) for statement in loaded)
    finally:
        path.unlink(missing_ok=True)


def test_generated_statement_counts_are_exact() -> None:
    path = Path("_generated_statement_counts_test.jsonl")
    try:
        statements = generate_and_save_labeled_statement_counts(
            true_count=5,
            false_count=7,
            output_path=path,
            max_depth=2,
            seed=11,
        )
        loaded = load_labeled_statements(path)

        assert len(statements) == 12
        assert sum(1 for statement in loaded if statement.label) == 5
        assert sum(1 for statement in loaded if not statement.label) == 7
        assert all(check_statement_truth(statement.formula) == statement.label for statement in loaded)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_formula_codecs_round_trip()
    test_truth_dataset_labels_are_checked()
    test_hash_embedding_shape_and_stability()
    test_generated_statement_file_round_trip()
    test_generated_statement_counts_are_exact()
    print("tree encoding checks passed")
