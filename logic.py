from __future__ import annotations
from dataclasses import dataclass
from typing import Union


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