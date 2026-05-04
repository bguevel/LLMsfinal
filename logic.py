from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Union


@dataclass(frozen=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class And:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} /\\ {self.right})"


@dataclass(frozen=True)
class Or:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} \\/ {self.right})"


@dataclass(frozen=True)
class Imp:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"


Formula = Union[Var, And, Or, Imp]


def pretty_formula(f: Formula) -> str:
    return str(f)


def formula_to_dict(f: Formula) -> dict:
    if isinstance(f, Var):
        return {
            "type": "var",
            "name": f.name,
        }

    if isinstance(f, And):
        return {
            "type": "and",
            "left": formula_to_dict(f.left),
            "right": formula_to_dict(f.right),
        }

    if isinstance(f, Or):
        return {
            "type": "or",
            "left": formula_to_dict(f.left),
            "right": formula_to_dict(f.right),
        }

    if isinstance(f, Imp):
        return {
            "type": "imp",
            "left": formula_to_dict(f.left),
            "right": formula_to_dict(f.right),
        }

    raise TypeError(f"Unknown formula type: {type(f)}")


def formula_from_dict(data: dict) -> Formula:
    kind = data.get("type")

    if kind == "var":
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError("Variable formula must include a string name")
        return Var(name)

    if kind in {"and", "or", "imp"}:
        left = formula_from_dict(data["left"])
        right = formula_from_dict(data["right"])

        if kind == "and":
            return And(left, right)
        if kind == "or":
            return Or(left, right)
        if kind == "imp":
            return Imp(left, right)

    raise ValueError(f"Unknown formula dict type: {kind!r}")


def formula_variables(f: Formula) -> List[str]:
    variables: set[str] = set()

    def visit(node: Formula) -> None:
        if isinstance(node, Var):
            variables.add(node.name)
            return

        if isinstance(node, (And, Or, Imp)):
            visit(node.left)
            visit(node.right)
            return

        raise TypeError(f"Unknown formula type: {type(node)}")

    visit(f)
    return sorted(variables)


def evaluate_formula(f: Formula, assignment: Dict[str, bool]) -> bool:
    if isinstance(f, Var):
        if f.name not in assignment:
            raise KeyError(f"Missing truth assignment for variable {f.name!r}")
        return assignment[f.name]

    if isinstance(f, And):
        return evaluate_formula(f.left, assignment) and evaluate_formula(f.right, assignment)

    if isinstance(f, Or):
        return evaluate_formula(f.left, assignment) or evaluate_formula(f.right, assignment)

    if isinstance(f, Imp):
        return (not evaluate_formula(f.left, assignment)) or evaluate_formula(f.right, assignment)

    raise TypeError(f"Unknown formula type: {type(f)}")


def all_truth_assignments(variables: Iterable[str]) -> List[Dict[str, bool]]:
    names = list(variables)
    return [
        dict(zip(names, values))
        for values in product([False, True], repeat=len(names))
    ]


def truth_table(f: Formula) -> List[tuple[Dict[str, bool], bool]]:
    return [
        (assignment, evaluate_formula(f, assignment))
        for assignment in all_truth_assignments(formula_variables(f))
    ]


def is_tautology(f: Formula) -> bool:
    return all(value for _, value in truth_table(f))


def is_satisfiable(f: Formula) -> bool:
    return any(value for _, value in truth_table(f))
