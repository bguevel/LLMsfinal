from __future__ import annotations

import argparse
import json
import random
import re
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
VARIABLE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
MAX_VARIABLES_PER_STATEMENT = 4


@dataclass(frozen=True)
class StatementComplexity:
    key: str
    label: str
    max_depth: int
    variables: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class StatementGenerationBatch:
    label: str
    total_count: int
    max_depth: int
    variables: tuple[str, ...]
    true_fraction: float = 0.5


@dataclass(frozen=True)
class StatementGenerationConfig:
    statements_per_level: int
    complexity_levels: int
    variables: tuple[str, ...]
    true_fraction: float = 0.5


STATEMENT_COMPLEXITY_LEVELS = (
    StatementComplexity(
        key="simple",
        label="Simple",
        max_depth=1,
        variables=("P", "Q"),
        description="short statements over two variables",
    ),
    StatementComplexity(
        key="moderate",
        label="Moderate",
        max_depth=3,
        variables=DEFAULT_VARIABLES,
        description="nested statements with the default variable set",
    ),
    StatementComplexity(
        key="complex",
        label="Complex",
        max_depth=5,
        variables=("P", "Q", "R", "S", "U", "V"),
        description="deeper statements with more variable variety",
    ),
)
DEFAULT_STATEMENT_COMPLEXITY = "moderate"
_COMPLEXITY_ALIASES = {
    "easy": "simple",
    "basic": "simple",
    "medium": "moderate",
    "normal": "moderate",
    "hard": "complex",
    "difficult": "complex",
}


def get_statement_complexity(level: str) -> StatementComplexity:
    normalized = level.strip().lower()
    normalized = _COMPLEXITY_ALIASES.get(normalized, normalized)

    for option in STATEMENT_COMPLEXITY_LEVELS:
        if option.key == normalized:
            return option

    valid = ", ".join(option.key for option in STATEMENT_COMPLEXITY_LEVELS)
    raise ValueError(f"Unknown statement complexity {level!r}. Choose one of: {valid}")


def _parse_complexity_key(raw: str) -> str:
    try:
        return get_statement_complexity(raw).key
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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


def split_label_counts(n: int, true_fraction: float = 0.5) -> tuple[int, int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= true_fraction <= 1.0:
        raise ValueError("true_fraction must be between 0 and 1")

    true_count = int((n * true_fraction) + 0.5)
    false_count = n - true_count
    return true_count, false_count


def _parse_first_int(raw: str) -> int | None:
    match = re.search(r"-?\d+", raw)
    return int(match.group(0)) if match else None


def _parse_first_float(raw: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    return float(match.group(0)) if match else None


def _parse_variable_set(raw: str) -> tuple[str, ...]:
    cleaned = raw.replace("{", " ").replace("}", " ").replace(",", " ")
    variables: list[str] = []
    seen: set[str] = set()
    for item in (part.strip() for part in cleaned.split()):
        if not item:
            continue
        if not VARIABLE_NAME_RE.fullmatch(item):
            raise ValueError(
                "Variable names must start with a letter and contain only "
                f"letters, digits, or underscores: {item!r}"
            )
        if item not in seen:
            variables.append(item)
            seen.add(item)
    if not variables:
        raise ValueError("set of variables must contain at least one variable name")
    return tuple(variables)


def _generation_config_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    variable_key: str | None = None
    variable_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if variable_key is not None:
            variable_lines.append(line)
            if "}" in line:
                entries.append((variable_key, " ".join(variable_lines)))
                variable_key = None
                variable_lines = []
            continue

        if ":" not in line:
            continue

        raw_key, raw_value = line.split(":", 1)
        key = raw_key.strip().lower()
        value = raw_value.strip()

        if "variable" in key and "{" in value and "}" not in value:
            variable_key = key
            variable_lines = [value]
            continue

        entries.append((key, value))

    if variable_key is not None:
        entries.append((variable_key, " ".join(variable_lines)))

    return entries


def parse_generation_config_text(text: str) -> StatementGenerationConfig:
    statements_per_level: int | None = None
    complexity_levels: int | None = None
    variables: tuple[str, ...] | None = None
    true_fraction = 0.5

    for key, value in _generation_config_entries(text):
        if "variable" in key:
            variables = _parse_variable_set(value)
        elif "fraction" in key and "true" in key:
            parsed_fraction = _parse_first_float(value)
            if parsed_fraction is None:
                raise ValueError(f"Could not parse true fraction from value: {value}")
            true_fraction = parsed_fraction
        elif "level" in key and "complex" in key:
            parsed_levels = _parse_first_int(value)
            if parsed_levels is None:
                raise ValueError(f"Could not parse complexity level count from value: {value}")
            complexity_levels = parsed_levels
        elif "number" in key or "(n)" in key:
            parsed_count = _parse_first_int(value)
            if parsed_count is None:
                raise ValueError(f"Could not parse statement count from value: {value}")
            statements_per_level = parsed_count

    if statements_per_level is None:
        raise ValueError("Missing statement count line, for example: number of true and false (n): 500")
    if complexity_levels is None:
        raise ValueError(
            "Missing complexity level line, for example: "
            "levels of complexity (for each level n statements are generated): 10"
        )
    if variables is None:
        raise ValueError("Missing variable set line, for example: set of variables: {P, Q, R}")
    if statements_per_level < 1:
        raise ValueError("statement count must be at least 1")
    if complexity_levels < 1:
        raise ValueError("complexity levels must be at least 1")
    if not 0.0 <= true_fraction <= 1.0:
        raise ValueError("true fraction must be between 0 and 1")

    return StatementGenerationConfig(
        statements_per_level=statements_per_level,
        complexity_levels=complexity_levels,
        variables=variables,
        true_fraction=true_fraction,
    )


def load_generation_config(path: str | Path) -> StatementGenerationConfig:
    config_path = Path(path)
    return parse_generation_config_text(config_path.read_text(encoding="utf-8"))


def statement_generation_batches_from_config(
    config: StatementGenerationConfig,
) -> list[StatementGenerationBatch]:
    return [
        StatementGenerationBatch(
            label=f"Level {level}",
            total_count=config.statements_per_level,
            max_depth=level,
            variables=config.variables,
            true_fraction=config.true_fraction,
        )
        for level in range(1, config.complexity_levels + 1)
    ]


def verified_statement_label(formula: Formula, expected_label: bool, name: str = "<unnamed>") -> bool:
    actual_label = check_statement_truth(formula)
    if actual_label != expected_label:
        raise ValueError(
            f"Truth label mismatch for {name}: expected {expected_label}, got {actual_label}"
        )
    return actual_label


def _random_variable(rng: random.Random, variables: Sequence[str]) -> Var:
    return Var(rng.choice(variables))


def _statement_variable_pool(
    variables: Sequence[str],
    rng: random.Random,
    max_variables: int = MAX_VARIABLES_PER_STATEMENT,
) -> tuple[str, ...]:
    if not variables:
        raise ValueError("variables must contain at least one variable name")
    if max_variables < 1:
        raise ValueError("max_variables must be at least 1")

    options = tuple(variables)
    if len(options) <= max_variables:
        return options
    return tuple(rng.sample(options, max_variables))


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
    true_count, false_count = split_label_counts(n, true_fraction)
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
    existing_seen: set[str] | None = None,
    name_prefix: str = "",
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
    seen = existing_seen if existing_seen is not None else set()
    attempts = 0
    max_attempts = max(1000, n * 200)

    while targets and attempts < max_attempts:
        attempts += 1
        label = targets[0]
        statement_variables = _statement_variable_pool(variables, rng)
        formula = (
            create_true_statement(max_depth=max_depth, variables=statement_variables, rng=rng)
            if label
            else create_false_statement(max_depth=max_depth, variables=statement_variables, rng=rng)
        )
        actual_label = check_statement_truth(formula)

        if actual_label != label:
            continue

        key = json.dumps(formula_to_dict(formula), sort_keys=True)
        if unique and key in seen:
            continue

        if unique:
            seen.add(key)
        index = len(statements)
        statements.append(
            LabeledStatement(
                name=f"{name_prefix}{'true' if label else 'false'}_{index:04d}",
                formula=formula,
                label=label,
            )
        )
        targets.pop(0)

    if targets:
        raise RuntimeError(f"Generated {len(statements)} statements but needed {n}")

    return statements


def _statement_name_fragment(raw: str) -> str:
    fragment = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    while "__" in fragment:
        fragment = fragment.replace("__", "_")
    return fragment.strip("_") or "level"


def generate_labeled_statement_batches(
    batches: Sequence[StatementGenerationBatch],
    seed: int | None = None,
    unique: bool = True,
    progress: bool = False,
) -> list[LabeledStatement]:
    if not batches:
        raise ValueError("batches must contain at least one complexity level")

    rng = random.Random(seed)
    seen: set[str] | None = set() if unique else None
    statements: list[LabeledStatement] = []

    for batch_index, batch in enumerate(batches, start=1):
        true_count, false_count = split_label_counts(batch.total_count, batch.true_fraction)
        if not batch.variables:
            raise ValueError("variables must contain at least one variable name")

        if true_count + false_count == 0:
            continue

        if progress:
            print(
                f"[generate] {batch.label}: total={batch.total_count}, "
                f"true={true_count}, false={false_count}, max_depth={batch.max_depth}, "
                f"variables={len(batch.variables)}, per-statement variable cap={MAX_VARIABLES_PER_STATEMENT}"
            )

        batch_seed = rng.randrange(0, 2**32)
        name_prefix = f"{_statement_name_fragment(batch.label)}_{batch_index:02d}_"
        batch_statements = generate_labeled_statement_counts(
            true_count=true_count,
            false_count=false_count,
            max_depth=batch.max_depth,
            variables=batch.variables,
            seed=batch_seed,
            unique=unique,
            existing_seen=seen,
            name_prefix=name_prefix,
        )
        statements.extend(batch_statements)
        if progress:
            print(f"[generate] {batch.label}: completed {len(batch_statements)} statements")

    rng.shuffle(statements)
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


def record_to_statement(record: dict, verify_label: bool = True) -> LabeledStatement:
    formula = formula_from_dict(record["formula"])
    label = bool(record["label"])
    if verify_label:
        actual_label = verified_statement_label(formula, label, str(record.get("name", "<unnamed>")))
        if "is_tautology" in record and bool(record["is_tautology"]) != actual_label:
            raise ValueError(
                f"Stored tautology metadata mismatch for {record.get('name', '<unnamed>')}: "
                f"expected {actual_label}, got {record['is_tautology']}"
            )
    elif "is_tautology" in record and bool(record["is_tautology"]) != label:
        raise ValueError(
            f"Stored label metadata mismatch for {record.get('name', '<unnamed>')}: "
            f"label={label}, is_tautology={record['is_tautology']}"
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


def load_labeled_statements(input_path: str | Path, verify_labels: bool = True) -> list[LabeledStatement]:
    path = Path(input_path)
    statements: list[LabeledStatement] = []

    with path.open("r", encoding="utf-8") as f:
        first = ""
        while True:
            char = f.read(1)
            if not char:
                return []
            if not char.isspace():
                first = char
                break
        f.seek(0)

        if first == "[":
            try:
                records = json.load(f)
            except Exception as exc:
                raise ValueError(f"Could not read JSON statement array from {path}") from exc
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array of statement records in {path}")
            for index, record in enumerate(records, start=1):
                try:
                    statements.append(record_to_statement(record, verify_label=verify_labels))
                except Exception as exc:
                    raise ValueError(f"Could not read statement array item {index}") from exc
            return statements

        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                statements.append(record_to_statement(json.loads(line), verify_label=verify_labels))
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


def generate_and_save_labeled_statement_batches(
    batches: Sequence[StatementGenerationBatch],
    output_path: str | Path,
    seed: int | None = None,
    unique: bool = True,
    progress: bool = False,
) -> list[LabeledStatement]:
    statements = generate_labeled_statement_batches(
        batches=batches,
        seed=seed,
        unique=unique,
        progress=progress,
    )
    save_labeled_statements(statements, output_path)
    return statements


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled propositional-logic statements.")
    parser.add_argument("n", type=int, help="Number of statements to generate.")
    parser.add_argument("output", type=Path, help="JSONL file to write.")
    parser.add_argument(
        "--complexity",
        type=_parse_complexity_key,
        default=DEFAULT_STATEMENT_COMPLEXITY,
        metavar="{simple,moderate,complex}",
        help="Named complexity preset for generated statements.",
    )
    parser.add_argument("--max-depth", type=int, default=None, help="Override the preset formula depth.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--true-fraction", type=float, default=0.5)
    parser.add_argument("--variables", nargs="+", default=None, help="Override the preset variable names.")
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    complexity = get_statement_complexity(args.complexity)
    max_depth = args.max_depth if args.max_depth is not None else complexity.max_depth
    variables = args.variables if args.variables is not None else complexity.variables
    statements = generate_and_save_labeled_statements(
        n=args.n,
        output_path=args.output,
        max_depth=max_depth,
        variables=variables,
        seed=args.seed,
        true_fraction=args.true_fraction,
        unique=not args.allow_duplicates,
    )

    true_count = sum(1 for statement in statements if statement.label)
    false_count = len(statements) - true_count
    print(f"Wrote {len(statements)} statements to {args.output}")
    print(
        f"Complexity: {complexity.label.lower()} "
        f"(max_depth={max_depth}, variables={', '.join(variables)})"
    )
    print(f"True: {true_count}")
    print(f"False: {false_count}")


if __name__ == "__main__":
    main()
