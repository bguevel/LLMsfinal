from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from logic import And, Formula, Imp, Or, Var
from state import Goal, ProofState


NODE_KINDS = ("var", "and", "or", "imp")
BINARY_KINDS = ("and", "or", "imp")


def formula_kind(formula: Formula) -> str:
    if isinstance(formula, Var):
        return "var"
    if isinstance(formula, And):
        return "and"
    if isinstance(formula, Or):
        return "or"
    if isinstance(formula, Imp):
        return "imp"
    raise TypeError(f"Unknown formula type: {type(formula)}")


def formula_children(formula: Formula) -> tuple[Formula, ...]:
    if isinstance(formula, Var):
        return ()
    if isinstance(formula, (And, Or, Imp)):
        return (formula.left, formula.right)
    raise TypeError(f"Unknown formula type: {type(formula)}")


def formula_to_prefix_tokens(formula: Formula) -> list[str]:
    """
    Prefix notation is compact, sequence-model friendly, and exactly reversible.

    Example:
      Imp(Var("P"), And(Var("Q"), Var("R")))
      -> ["imp", "var", "P", "and", "var", "Q", "var", "R"]
    """
    if isinstance(formula, Var):
        return ["var", formula.name]

    if isinstance(formula, And):
        return ["and", *formula_to_prefix_tokens(formula.left), *formula_to_prefix_tokens(formula.right)]

    if isinstance(formula, Or):
        return ["or", *formula_to_prefix_tokens(formula.left), *formula_to_prefix_tokens(formula.right)]

    if isinstance(formula, Imp):
        return ["imp", *formula_to_prefix_tokens(formula.left), *formula_to_prefix_tokens(formula.right)]

    raise TypeError(f"Unknown formula type: {type(formula)}")


def prefix_tokens_to_formula(tokens: Sequence[str]) -> Formula:
    def parse_at(index: int) -> tuple[Formula, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of prefix token stream")

        token = tokens[index]

        if token == "var":
            name_index = index + 1
            if name_index >= len(tokens):
                raise ValueError("Variable token must be followed by a variable name")
            return Var(tokens[name_index]), name_index + 1

        if token in BINARY_KINDS:
            left, next_index = parse_at(index + 1)
            right, next_index = parse_at(next_index)

            if token == "and":
                return And(left, right), next_index
            if token == "or":
                return Or(left, right), next_index
            if token == "imp":
                return Imp(left, right), next_index

        raise ValueError(f"Unknown prefix token: {token!r}")

    formula, next_index = parse_at(0)
    if next_index != len(tokens):
        extra = list(tokens[next_index:])
        raise ValueError(f"Unused prefix tokens after parse: {extra!r}")
    return formula


def formula_to_prefix_text(formula: Formula) -> str:
    return json.dumps(formula_to_prefix_tokens(formula))


def prefix_text_to_formula(text: str) -> Formula:
    tokens = json.loads(text)
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError("Prefix text must decode to a JSON list of strings")
    return prefix_tokens_to_formula(tokens)


def formula_to_sexpr(formula: Formula) -> str:
    """
    S-expressions are human readable and still keep the tree explicit.
    Variable names are JSON-quoted so the format round-trips safely.
    """
    if isinstance(formula, Var):
        return f"(var {json.dumps(formula.name)})"

    if isinstance(formula, And):
        return f"(and {formula_to_sexpr(formula.left)} {formula_to_sexpr(formula.right)})"

    if isinstance(formula, Or):
        return f"(or {formula_to_sexpr(formula.left)} {formula_to_sexpr(formula.right)})"

    if isinstance(formula, Imp):
        return f"(imp {formula_to_sexpr(formula.left)} {formula_to_sexpr(formula.right)})"

    raise TypeError(f"Unknown formula type: {type(formula)}")


def _tokenize_sexpr(text: str) -> Iterator[str]:
    decoder = json.JSONDecoder()
    index = 0

    while index < len(text):
        char = text[index]

        if char.isspace():
            index += 1
            continue

        if char in "()":
            yield char
            index += 1
            continue

        if char == '"':
            value, next_index = decoder.raw_decode(text, index)
            if not isinstance(value, str):
                raise ValueError("Quoted S-expression atoms must be strings")
            yield value
            index = next_index
            continue

        start = index
        while index < len(text) and not text[index].isspace() and text[index] not in "()":
            index += 1
        yield text[start:index]


def sexpr_to_formula(text: str) -> Formula:
    tokens = list(_tokenize_sexpr(text))

    def require(index: int, expected: str) -> int:
        if index >= len(tokens) or tokens[index] != expected:
            got = tokens[index] if index < len(tokens) else "<end>"
            raise ValueError(f"Expected {expected!r}, got {got!r}")
        return index + 1

    def parse_at(index: int) -> tuple[Formula, int]:
        index = require(index, "(")
        if index >= len(tokens):
            raise ValueError("Missing S-expression operator")

        op = tokens[index]
        index += 1

        if op == "var":
            if index >= len(tokens):
                raise ValueError("Missing variable name")
            name = tokens[index]
            index += 1
            index = require(index, ")")
            return Var(name), index

        if op in BINARY_KINDS:
            left, index = parse_at(index)
            right, index = parse_at(index)
            index = require(index, ")")

            if op == "and":
                return And(left, right), index
            if op == "or":
                return Or(left, right), index
            if op == "imp":
                return Imp(left, right), index

        raise ValueError(f"Unknown S-expression operator: {op!r}")

    formula, next_index = parse_at(0)
    if next_index != len(tokens):
        raise ValueError(f"Unused S-expression tokens after parse: {tokens[next_index:]!r}")
    return formula


@dataclass(frozen=True)
class TreePathEntry:
    path: str
    kind: str
    value: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        data: dict[str, str | None] = {
            "path": self.path,
            "kind": self.kind,
        }
        if self.value is not None:
            data["value"] = self.value
        return data


def formula_to_path_entries(formula: Formula, path: str = "root") -> list[TreePathEntry]:
    """
    Path encoding is good for graph/tree-aware models.

    Each node says what lives at root, root.L, root.R, etc. Because the complete
    map is retained, this is exactly reversible.
    """
    if isinstance(formula, Var):
        return [TreePathEntry(path=path, kind="var", value=formula.name)]

    if isinstance(formula, (And, Or, Imp)):
        kind = formula_kind(formula)
        return [
            TreePathEntry(path=path, kind=kind),
            *formula_to_path_entries(formula.left, f"{path}.L"),
            *formula_to_path_entries(formula.right, f"{path}.R"),
        ]

    raise TypeError(f"Unknown formula type: {type(formula)}")


def path_entries_to_formula(entries: Iterable[TreePathEntry | dict[str, str | None]]) -> Formula:
    table: dict[str, TreePathEntry] = {}

    for entry in entries:
        if isinstance(entry, TreePathEntry):
            item = entry
        else:
            value = entry.get("value")
            item = TreePathEntry(
                path=str(entry["path"]),
                kind=str(entry["kind"]),
                value=None if value is None else str(value),
            )
        if item.path in table:
            raise ValueError(f"Duplicate tree path: {item.path}")
        table[item.path] = item

    def build(path: str) -> Formula:
        if path not in table:
            raise ValueError(f"Missing tree path: {path}")

        item = table[path]

        if item.kind == "var":
            if item.value is None:
                raise ValueError(f"Variable at {path} is missing a name")
            return Var(item.value)

        if item.kind in BINARY_KINDS:
            left = build(f"{path}.L")
            right = build(f"{path}.R")
            if item.kind == "and":
                return And(left, right)
            if item.kind == "or":
                return Or(left, right)
            if item.kind == "imp":
                return Imp(left, right)

        raise ValueError(f"Unknown node kind at {path}: {item.kind!r}")

    return build("root")


def formula_to_path_json(formula: Formula) -> str:
    return json.dumps([entry.to_dict() for entry in formula_to_path_entries(formula)])


def path_json_to_formula(text: str) -> Formula:
    entries = json.loads(text)
    if not isinstance(entries, list):
        raise ValueError("Path encoding must decode to a JSON list")
    return path_entries_to_formula(entries)


def formula_tree_stats(formula: Formula) -> dict[str, int]:
    counts = {kind: 0 for kind in NODE_KINDS}

    def visit(node: Formula, depth: int) -> tuple[int, int]:
        kind = formula_kind(node)
        counts[kind] += 1
        children = formula_children(node)

        if not children:
            return 1, depth

        child_nodes = 1
        max_depth = depth
        for child in children:
            node_count, child_depth = visit(child, depth + 1)
            child_nodes += node_count
            max_depth = max(max_depth, child_depth)
        return child_nodes, max_depth

    nodes, max_depth = visit(formula, 0)
    leaves = counts["var"]

    return {
        "nodes": nodes,
        "leaves": leaves,
        "max_depth": max_depth,
        "vars": counts["var"],
        "ands": counts["and"],
        "ors": counts["or"],
        "imps": counts["imp"],
    }


def formula_to_tree_record(formula: Formula) -> dict:
    return {
        "prefix_tokens": formula_to_prefix_tokens(formula),
        "sexpr": formula_to_sexpr(formula),
        "paths": [entry.to_dict() for entry in formula_to_path_entries(formula)],
        "stats": formula_tree_stats(formula),
    }


def proof_state_to_tree_json(state: ProofState) -> dict:
    if state.is_solved():
        return {
            "status": "solved",
            "goals": [],
        }

    return {
        "status": "unsolved",
        "goals": [
            {
                "assumptions": [formula_to_tree_record(assumption) for assumption in goal.assumptions],
                "target": formula_to_tree_record(goal.target),
            }
            for goal in state.goals
        ],
    }


def proof_state_to_tree_text(state: ProofState) -> str:
    return json.dumps(proof_state_to_tree_json(state), indent=2)


def _stable_hash_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vec)
    if norm.item() == 0:
        return vec
    return vec / norm


class HashingTreeEmbedder:
    """
    Fixed structural embedding for formulas/proof states.

    This is useful before training because it does not introduce random weights.
    Features include node kinds at paths, variable names at paths, operator edges,
    prefix-token ngrams, and basic tree stats.
    """

    def __init__(self, dim: int = 256):
        if torch is None:
            raise ModuleNotFoundError("HashingTreeEmbedder requires torch")
        if dim < 32:
            raise ValueError("HashingTreeEmbedder dim should be at least 32")
        self.dim = dim

    def _add(self, vec: torch.Tensor, feature: str, weight: float = 1.0) -> None:
        raw = _stable_hash_int(feature)
        index = raw % self.dim
        sign = 1.0 if ((raw >> 63) & 1) else -1.0
        vec[index] += sign * weight

    def _formula_features(self, formula: Formula, prefix: str) -> list[tuple[str, float]]:
        features: list[tuple[str, float]] = []
        entries = formula_to_path_entries(formula)
        by_path = {entry.path: entry for entry in entries}

        for entry in entries:
            features.append((f"{prefix}:node:{entry.path}:{entry.kind}", 1.0))
            if entry.value is not None:
                features.append((f"{prefix}:var:{entry.path}:{entry.value}", 1.0))

            if entry.path != "root":
                parent_path, side = entry.path.rsplit(".", 1)
                parent = by_path[parent_path]
                features.append((f"{prefix}:edge:{parent.kind}:{side}:{entry.kind}", 1.0))

        tokens = formula_to_prefix_tokens(formula)
        for token in tokens:
            features.append((f"{prefix}:prefix1:{token}", 0.5))
        for left, right in zip(tokens, tokens[1:]):
            features.append((f"{prefix}:prefix2:{left}>{right}", 0.75))

        for name, value in formula_tree_stats(formula).items():
            features.append((f"{prefix}:stat:{name}", float(value)))

        return features

    def encode_formula(self, formula: Formula, role: str = "formula") -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        for feature, weight in self._formula_features(formula, role):
            self._add(vec, feature, weight)
        return _l2_normalize(vec)

    def encode_goal(self, goal: Goal) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        for feature, weight in self._formula_features(goal.target, "target"):
            self._add(vec, feature, weight)

        for assumption in goal.assumptions:
            for feature, weight in self._formula_features(assumption, "assumption"):
                self._add(vec, feature, weight)

        self._add(vec, f"goal:assumption_count:{len(goal.assumptions)}", float(len(goal.assumptions)))
        return _l2_normalize(vec)

    def encode_state(self, state: ProofState) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        if state.is_solved():
            self._add(vec, "state:solved")
            return _l2_normalize(vec)

        for goal_index, goal in enumerate(state.goals):
            goal_vec = self.encode_goal(goal)
            vec += goal_vec / float(goal_index + 1)

        self._add(vec, f"state:goal_count:{len(state.goals)}", float(len(state.goals)))
        return _l2_normalize(vec)


if nn is None:

    class FormulaTreeEncoder:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("FormulaTreeEncoder requires torch")

else:

    class FormulaTreeEncoder(nn.Module):
        """
        Trainable recursive encoder for future experiments.

        It produces one vector per formula by recursively combining operator,
        left-child, and right-child embeddings. Decoding from this continuous vector
        would need a trained decoder head; use the exact codecs above for lossless
        unembedding during data generation and debugging.
        """

        def __init__(self, dim: int = 128, var_buckets: int = 1024):
            super().__init__()
            self.dim = dim
            self.var_buckets = var_buckets
            self.kind_to_index = {kind: i for i, kind in enumerate(NODE_KINDS)}

            self.kind_embedding = nn.Embedding(len(NODE_KINDS), dim)
            self.var_embedding = nn.Embedding(var_buckets, dim)
            self.leaf = nn.Linear(dim * 2, dim)
            self.binary = nn.Linear(dim * 3, dim)
            self.role_embedding = nn.Embedding(2, dim)
            self.state_projection = nn.Linear(dim, dim)

        def _kind_tensor(self, kind: str, device: torch.device) -> torch.Tensor:
            return torch.tensor(self.kind_to_index[kind], dtype=torch.long, device=device)

        def _var_tensor(self, name: str, device: torch.device) -> torch.Tensor:
            bucket = _stable_hash_int(name) % self.var_buckets
            return torch.tensor(bucket, dtype=torch.long, device=device)

        def encode_formula(self, formula: Formula, device: torch.device | None = None) -> torch.Tensor:
            if device is None:
                device = self.kind_embedding.weight.device

            kind = formula_kind(formula)
            kind_vec = self.kind_embedding(self._kind_tensor(kind, device))

            if isinstance(formula, Var):
                var_vec = self.var_embedding(self._var_tensor(formula.name, device))
                return torch.tanh(self.leaf(torch.cat([kind_vec, var_vec], dim=-1)))

            left_vec = self.encode_formula(formula.left, device=device)
            right_vec = self.encode_formula(formula.right, device=device)
            return torch.tanh(self.binary(torch.cat([kind_vec, left_vec, right_vec], dim=-1)))

        def encode_goal(self, goal: Goal, device: torch.device | None = None) -> torch.Tensor:
            if device is None:
                device = self.kind_embedding.weight.device

            target_role = self.role_embedding(torch.tensor(0, dtype=torch.long, device=device))
            assumption_role = self.role_embedding(torch.tensor(1, dtype=torch.long, device=device))

            pieces = [self.encode_formula(goal.target, device=device) + target_role]
            pieces.extend(self.encode_formula(assumption, device=device) + assumption_role for assumption in goal.assumptions)

            return torch.stack(pieces).mean(dim=0)

        def encode_state(self, state: ProofState, device: torch.device | None = None) -> torch.Tensor:
            if device is None:
                device = self.kind_embedding.weight.device

            if state.is_solved():
                return torch.zeros(self.dim, device=device)

            goal_vecs = [self.encode_goal(goal, device=device) for goal in state.goals]
            return torch.tanh(self.state_projection(torch.stack(goal_vecs).mean(dim=0)))
