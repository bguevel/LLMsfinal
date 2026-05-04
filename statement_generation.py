from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from logic import (
    And,
    Formula,
    Imp,
    Or,
    Var,
    formula_from_dict,
    formula_to_dict,
    is_satisfiable,
    is_tautology,
)
from tree_encoding import formula_to_tree_record


DEFAULT_VARIABLES = ("P", "Q", "R", "S")
STATEMENT_QUESTION_PREFIX = "is this statement true"


def statement_text_for_formula(formula: Formula) -> str:
    return f"{STATEMENT_QUESTION_PREFIX} {formula}"


@dataclass(frozen=True)
class LabeledStatement:
    name: str
    formula: Formula
    label: bool

    @property
    def text(self) -> str:
        return statement_text_for_formula(self.formula)


def check_statement_truth(formula: Formula) -> bool:
    """
    Return True when a formula is true under every truth assignment.

    In this propositional framework, "false" examples are formulas that are not
    tautologies. They may still be satisfiable under some assignments.
    """
    return is_tautology(formula)


def verified_statement_label(formula: Formula, expected_label: bool, name: str = "<unnamed>") -> bool:
    actual_label = check_statement_truth(formula)
    if actual_label != expected_label:
        raise ValueError(
            f"Truth label mismatch for {name}: expected {expected_label}, got {actual_label}"
        )
    return actual_label


def _random_variable(rng: random.Random, variables: Sequence[str]) -> Var:
    return Var(rng.choice(variables))


def random_formula(
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    rng: random.Random | None = None,
) -> Formula:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if not variables:
        raise ValueError("variables must contain at least one variable name")

    rng = rng or random.Random()

    if max_depth == 0 or rng.random() < 0.25:
        return _random_variable(rng, variables)

    left = random_formula(max_depth - 1, variables, rng)
    right = random_formula(max_depth - 1, variables, rng)
    op = rng.choice((And, Or, Imp))
    return op(left, right)


def create_true_statement(
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    rng: random.Random | None = None,
) -> Formula:
    rng = rng or random.Random()
    a = random_formula(max_depth, variables, rng)
    b = random_formula(max_depth, variables, rng)
    c = random_formula(max_depth, variables, rng)

    templates = [
        lambda: Imp(a, a),
        lambda: Imp(a, Imp(b, a)),
        lambda: Imp(And(a, b), a),
        lambda: Imp(And(a, b), b),
        lambda: Imp(a, Or(a, b)),
        lambda: Imp(b, Or(a, b)),
        lambda: Imp(And(a, Imp(a, b)), b),
        lambda: Imp(Imp(a, b), Imp(Imp(b, c), Imp(a, c))),
        lambda: Imp(Or(a, b), Or(b, a)),
        lambda: Imp(And(a, b), And(b, a)),
    ]

    for _ in range(100):
        formula = rng.choice(templates)()
        if check_statement_truth(formula):
            return formula

    raise RuntimeError("Could not generate a true statement")


def create_false_statement(
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    rng: random.Random | None = None,
) -> Formula:
    rng = rng or random.Random()

    false_templates = [
        lambda: _random_variable(rng, variables),
        lambda: And(random_formula(max_depth, variables, rng), random_formula(max_depth, variables, rng)),
        lambda: Or(_random_variable(rng, variables), _random_variable(rng, variables)),
        lambda: Imp(_random_variable(rng, variables), _random_variable(rng, variables)),
        lambda: Imp(
            random_formula(max_depth, variables, rng),
            And(random_formula(max_depth, variables, rng), random_formula(max_depth, variables, rng)),
        ),
    ]

    for _ in range(500):
        if rng.random() < 0.7:
            formula = rng.choice(false_templates)()
        else:
            formula = random_formula(max_depth, variables, rng)

        if not check_statement_truth(formula):
            return formula

    raise RuntimeError("Could not generate a false statement")


def generate_labeled_statements(
    n: int,
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    seed: int | None = None,
    true_fraction: float = 0.5,
    unique: bool = True,
) -> list[LabeledStatement]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= true_fraction <= 1.0:
        raise ValueError("true_fraction must be between 0 and 1")

    true_count = int((n * true_fraction) + 0.5)
    false_count = n - true_count
    return generate_labeled_statement_counts(
        true_count=true_count,
        false_count=false_count,
        max_depth=max_depth,
        variables=variables,
        seed=seed,
        unique=unique,
    )


def generate_labeled_statement_counts(
    true_count: int,
    false_count: int,
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    seed: int | None = None,
    unique: bool = True,
) -> list[LabeledStatement]:
    if true_count < 0:
        raise ValueError("true_count must be non-negative")
    if false_count < 0:
        raise ValueError("false_count must be non-negative")

    n = true_count + false_count
    rng = random.Random(seed)
    targets = [True] * true_count + [False] * false_count
    rng.shuffle(targets)

    statements: list[LabeledStatement] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(1000, n * 200)

    while targets and attempts < max_attempts:
        attempts += 1
        label = targets[0]
        formula = (
            create_true_statement(max_depth=max_depth, variables=variables, rng=rng)
            if label
            else create_false_statement(max_depth=max_depth, variables=variables, rng=rng)
        )
        actual_label = check_statement_truth(formula)

        if actual_label != label:
            continue

        key = json.dumps(formula_to_dict(formula), sort_keys=True)
        if unique and key in seen:
            continue

        seen.add(key)
        index = len(statements)
        statements.append(
            LabeledStatement(
                name=f"{'true' if label else 'false'}_{index:04d}",
                formula=formula,
                label=label,
            )
        )
        targets.pop(0)

    if targets:
        raise RuntimeError(f"Generated {len(statements)} statements but needed {n}")

    return statements


def statement_to_record(statement: LabeledStatement) -> dict:
    actual_label = verified_statement_label(statement.formula, statement.label, statement.name)
    return {
        "name": statement.name,
        "label": statement.label,
        "is_tautology": actual_label,
        "is_satisfiable": is_satisfiable(statement.formula),
        "text": statement.text,
        "formula": formula_to_dict(statement.formula),
        "tree": formula_to_tree_record(statement.formula),
    }


def record_to_statement(record: dict) -> LabeledStatement:
    formula = formula_from_dict(record["formula"])
    label = bool(record["label"])
    actual_label = verified_statement_label(formula, label, str(record.get("name", "<unnamed>")))
    if "is_tautology" in record and bool(record["is_tautology"]) != actual_label:
        raise ValueError(
            f"Stored tautology metadata mismatch for {record.get('name', '<unnamed>')}: "
            f"expected {actual_label}, got {record['is_tautology']}"
        )

    return LabeledStatement(
        name=str(record["name"]),
        formula=formula,
        label=label,
    )


def save_labeled_statements(statements: Iterable[LabeledStatement], output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for statement in statements:
            f.write(json.dumps(statement_to_record(statement), sort_keys=True))
            f.write("\n")

    return path


def load_labeled_statements(input_path: str | Path) -> list[LabeledStatement]:
    path = Path(input_path)
    statements: list[LabeledStatement] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                statements.append(record_to_statement(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Could not read statement on line {line_number}") from exc

    return statements


def generate_and_save_labeled_statements(
    n: int,
    output_path: str | Path,
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    seed: int | None = None,
    true_fraction: float = 0.5,
    unique: bool = True,
) -> list[LabeledStatement]:
    statements = generate_labeled_statements(
        n=n,
        max_depth=max_depth,
        variables=variables,
        seed=seed,
        true_fraction=true_fraction,
        unique=unique,
    )
    save_labeled_statements(statements, output_path)
    return statements


def generate_and_save_labeled_statement_counts(
    true_count: int,
    false_count: int,
    output_path: str | Path,
    max_depth: int = 3,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    seed: int | None = None,
    unique: bool = True,
) -> list[LabeledStatement]:
    statements = generate_labeled_statement_counts(
        true_count=true_count,
        false_count=false_count,
        max_depth=max_depth,
        variables=variables,
        seed=seed,
        unique=unique,
    )
    save_labeled_statements(statements, output_path)
    return statements


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled propositional-logic statements.")
    parser.add_argument("n", type=int, help="Number of statements to generate.")
    parser.add_argument("output", type=Path, help="JSONL file to write.")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--true-fraction", type=float, default=0.5)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    statements = generate_and_save_labeled_statements(
        n=args.n,
        output_path=args.output,
        max_depth=args.max_depth,
        variables=args.variables,
        seed=args.seed,
        true_fraction=args.true_fraction,
        unique=not args.allow_duplicates,
    )

    true_count = sum(1 for statement in statements if statement.label)
    false_count = len(statements) - true_count
    print(f"Wrote {len(statements)} statements to {args.output}")
    print(f"True: {true_count}")
    print(f"False: {false_count}")


if __name__ == "__main__":
    main()
