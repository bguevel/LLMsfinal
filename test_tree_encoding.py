from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from dataset import make_truth_dataset
from statement_generation import (
    check_statement_truth,
    DEFAULT_STATEMENT_COMPLEXITY,
    StatementGenerationBatch,
    generate_and_save_labeled_statement_batches,
    generate_and_save_labeled_statement_counts,
    generate_and_save_labeled_statements,
    get_statement_complexity,
    load_labeled_statements,
    parse_generation_config_text,
    statement_text_for_formula,
    statement_generation_batches_from_config,
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
    for _, formula, _ in make_truth_dataset():
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
    _, formula, _ = make_truth_dataset()[0]

    first = embedder.encode_formula(formula)
    second = embedder.encode_formula(formula)

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


def test_generated_statement_batches_keep_complexity_counts() -> None:
    path = Path("_generated_statement_batches_test.jsonl")
    try:
        statements = generate_and_save_labeled_statement_batches(
            batches=[
                StatementGenerationBatch(
                    label="Simple",
                    total_count=4,
                    max_depth=1,
                    variables=("P", "Q"),
                ),
                StatementGenerationBatch(
                    label="Moderate",
                    total_count=6,
                    max_depth=2,
                    variables=("P", "Q", "R"),
                ),
            ],
            output_path=path,
            seed=17,
        )
        loaded = load_labeled_statements(path)

        assert len(statements) == 10
        assert len(loaded) == 10
        assert sum(1 for statement in loaded if statement.name.startswith("simple_01_")) == 4
        assert sum(1 for statement in loaded if statement.name.startswith("moderate_02_")) == 6
        assert sum(1 for statement in loaded if statement.label) == 5
        assert sum(1 for statement in loaded if not statement.label) == 5
        assert all(check_statement_truth(statement.formula) == statement.label for statement in loaded)
    finally:
        path.unlink(missing_ok=True)


def test_generation_config_text_creates_depth_batches() -> None:
    config = parse_generation_config_text(
        "number of true and false (n): 500\n"
        "levels of complexity (for each level n statements are generated): 10\n"
        "set of variables: {Q, W, E, R, T, Y, U, I, O, P}\n"
    )
    batches = statement_generation_batches_from_config(config)

    assert config.statements_per_level == 500
    assert config.complexity_levels == 10
    assert config.variables == ("Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P")
    assert len(batches) == 10
    assert batches[0].label == "Level 1"
    assert batches[0].max_depth == 1
    assert batches[-1].label == "Level 10"
    assert batches[-1].max_depth == 10
    assert all(batch.total_count == 500 for batch in batches)


def test_fast_statement_load_trusts_saved_generation_metadata() -> None:
    path = Path("_generated_statement_fast_load_test.jsonl")
    try:
        path.write_text(
            '{"formula": {"name": "P", "type": "var"}, "is_tautology": true, '
            '"label": true, "name": "bad_external_record"}\n',
            encoding="utf-8",
        )

        loaded = load_labeled_statements(path, verify_labels=False)
        assert len(loaded) == 1
        assert loaded[0].label is True

        try:
            load_labeled_statements(path)
        except ValueError:
            pass
        else:
            raise AssertionError("verified load should reject symbolic label mismatch")
    finally:
        path.unlink(missing_ok=True)


def test_statement_loader_accepts_json_arrays() -> None:
    path = Path("_generated_statement_array_test.json")
    try:
        path.write_text(
            '[{"formula": {"left": {"name": "P", "type": "var"}, '
            '"right": {"name": "P", "type": "var"}, "type": "imp"}, '
            '"is_tautology": true, "label": true, "name": "array_record"}]',
            encoding="utf-8",
        )

        loaded = load_labeled_statements(path, verify_labels=False)

        assert len(loaded) == 1
        assert loaded[0].name == "array_record"
        assert loaded[0].label is True
    finally:
        path.unlink(missing_ok=True)


def test_statement_complexity_presets_resolve() -> None:
    default = get_statement_complexity(DEFAULT_STATEMENT_COMPLEXITY)
    simple = get_statement_complexity("easy")
    complex_level = get_statement_complexity("hard")

    assert default.key == "moderate"
    assert simple.key == "simple"
    assert complex_level.key == "complex"
    assert simple.max_depth < default.max_depth < complex_level.max_depth


if __name__ == "__main__":
    test_formula_codecs_round_trip()
    test_truth_dataset_labels_are_checked()
    test_hash_embedding_shape_and_stability()
    test_generated_statement_file_round_trip()
    test_generated_statement_counts_are_exact()
    test_generated_statement_batches_keep_complexity_counts()
    test_generation_config_text_creates_depth_batches()
    test_fast_statement_load_trusts_saved_generation_metadata()
    test_statement_loader_accepts_json_arrays()
    test_statement_complexity_presets_resolve()
    print("tree encoding checks passed")
