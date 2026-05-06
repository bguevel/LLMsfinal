from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from logic import Formula
from statement_generation import LabeledStatement, check_statement_truth
from tree_encoding import FormulaTreeEncoder


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ModuleNotFoundError("truth_model requires torch")


if nn is None:

    class TruthEmbeddingClassifier:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("TruthEmbeddingClassifier requires torch")

else:

    class TruthEmbeddingClassifier(nn.Module):
        """
        FormulaTreeEncoder plus a truth/false classifier head.

        This is the trainable embedding path: the tree encoder learns formula
        vectors directly from truth labels, then the linear head predicts whether
        the statement is a tautology.
        """

        def __init__(self, dim: int = 128, var_buckets: int = 1024):
            super().__init__()
            self.dim = dim
            self.var_buckets = var_buckets
            self.encoder = FormulaTreeEncoder(dim=dim, var_buckets=var_buckets)
            self.classifier = nn.Linear(dim, 2)

        def encode_formula(self, formula: Formula) -> torch.Tensor:
            return self.encoder.encode_formula(formula)

        def forward_formula(self, formula: Formula) -> torch.Tensor:
            embedding = self.encode_formula(formula)
            return self.classifier(embedding)


def train_truth_embedding_model(
    statements: Sequence[LabeledStatement],
    dim: int = 128,
    var_buckets: int = 1024,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
    grad_clip: float | None = 1.0,
    seed: int | None = None,
    device: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[TruthEmbeddingClassifier, list[float]]:
    _require_torch()

    if not statements:
        raise ValueError("statements must not be empty")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TruthEmbeddingClassifier(dim=dim, var_buckets=var_buckets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data = list(statements)
    losses: list[float] = []

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        total_items = 0

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            logits = torch.stack([model.forward_formula(item.formula) for item in batch])
            labels = torch.tensor([1 if item.label else 0 for item in batch], dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_items += len(batch)

        avg_loss = total_loss / max(1, total_items)
        losses.append(avg_loss)
        epoch_number = epoch + 1
        if epoch_number % 50 == 0 or epoch_number == epochs:
            print(f"epoch {epoch_number}/{epochs} | truth embedding loss {avg_loss:.4f}")

    if save_path is not None:
        save_truth_embedding_model(model, save_path)

    return model, losses


@torch.no_grad() if torch is not None else (lambda fn: fn)
def predict_truth_with_embedding(model: TruthEmbeddingClassifier, formula: Formula) -> tuple[bool, float]:
    _require_torch()
    model.eval()
    logits = model.forward_formula(formula)
    probs = torch.softmax(logits, dim=-1)
    true_probability = float(probs[1].detach().cpu().item())
    return true_probability >= 0.5, true_probability


def evaluate_truth_embedding_model(
    model: TruthEmbeddingClassifier,
    statements: Sequence[LabeledStatement],
) -> dict[str, float]:
    _require_torch()
    if not statements:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
        }

    correct = 0
    for item in statements:
        prediction, _ = predict_truth_with_embedding(model, item.formula)
        correct += int(prediction == item.label)

    return {
        "accuracy": correct / len(statements),
        "correct": correct,
        "total": len(statements),
    }


def save_truth_embedding_model(model: TruthEmbeddingClassifier, path: str | Path) -> Path:
    _require_torch()
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "dim": model.dim,
            "var_buckets": model.var_buckets,
            "state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path


def load_truth_embedding_model(path: str | Path, device: str | None = None) -> TruthEmbeddingClassifier:
    _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model = TruthEmbeddingClassifier(
        dim=int(checkpoint["dim"]),
        var_buckets=int(checkpoint["var_buckets"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def verify_symbolic_labels(statements: Sequence[LabeledStatement]) -> tuple[int, int]:
    correct = 0
    for item in statements:
        correct += int(check_statement_truth(item.formula) == item.label)
    return correct, len(statements)
