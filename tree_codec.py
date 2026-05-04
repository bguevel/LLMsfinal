from __future__ import annotations

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
from statement_generation import LabeledStatement
from tree_encoding import FormulaTreeEncoder, formula_to_prefix_tokens, prefix_tokens_to_formula


PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ModuleNotFoundError("tree_codec requires torch")


def build_tree_token_vocab(formulas: Sequence[Formula]) -> dict[str, int]:
    vocab = {
        PAD: 0,
        BOS: 1,
        EOS: 2,
    }

    for formula in formulas:
        for token in formula_to_prefix_tokens(formula):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


if nn is None:

    class TrainableTreeCodec:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("TrainableTreeCodec requires torch")

else:

    class TrainableTreeCodec(nn.Module):
        """
        Trainable embedding/unembedding for formula trees.

        The encoder maps an AST to a continuous vector. The decoder tries to
        reconstruct the lossless prefix-token representation from that vector.
        """

        def __init__(self, vocab: dict[str, int], dim: int = 128, var_buckets: int = 1024):
            super().__init__()
            self.vocab = dict(vocab)
            self.itos = {i: token for token, i in self.vocab.items()}
            self.dim = dim
            self.var_buckets = var_buckets

            self.encoder = FormulaTreeEncoder(dim=dim, var_buckets=var_buckets)
            self.token_embedding = nn.Embedding(len(self.vocab), dim)
            self.decoder = nn.GRUCell(dim, dim)
            self.output = nn.Linear(dim, len(self.vocab))

        def _ids_for_formula(self, formula: Formula, device: torch.device) -> torch.Tensor:
            tokens = [BOS, *formula_to_prefix_tokens(formula), EOS]
            return torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long, device=device)

        def encode_formula(self, formula: Formula) -> torch.Tensor:
            return self.encoder.encode_formula(formula)

        def forward_formula(self, formula: Formula) -> tuple[torch.Tensor, torch.Tensor]:
            device = self.token_embedding.weight.device
            ids = self._ids_for_formula(formula, device=device)
            hidden = self.encode_formula(formula)

            logits = []
            for token_id in ids[:-1]:
                token_vec = self.token_embedding(token_id)
                hidden = self.decoder(token_vec, hidden)
                logits.append(self.output(hidden))

            return torch.stack(logits), ids[1:]

        @torch.no_grad()
        def decode_tokens(self, formula: Formula, max_tokens: int = 128) -> list[str]:
            device = self.token_embedding.weight.device
            hidden = self.encode_formula(formula)
            token_id = torch.tensor(self.vocab[BOS], dtype=torch.long, device=device)
            decoded: list[str] = []

            for _ in range(max_tokens):
                token_vec = self.token_embedding(token_id)
                hidden = self.decoder(token_vec, hidden)
                next_id = int(torch.argmax(self.output(hidden), dim=-1).detach().cpu().item())
                token = self.itos[next_id]

                if token == EOS:
                    break
                if token not in {PAD, BOS}:
                    decoded.append(token)

                token_id = torch.tensor(next_id, dtype=torch.long, device=device)

            return decoded

        @torch.no_grad()
        def decode_formula(self, formula: Formula, max_tokens: int = 128) -> Formula:
            return prefix_tokens_to_formula(self.decode_tokens(formula, max_tokens=max_tokens))


def train_tree_codec(
    statements: Sequence[LabeledStatement],
    dim: int = 128,
    var_buckets: int = 1024,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 8,
    grad_clip: float | None = 1.0,
    device: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[TrainableTreeCodec, list[float]]:
    _require_torch()

    if not statements:
        raise ValueError("statements must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_tree_token_vocab([item.formula for item in statements])
    model = TrainableTreeCodec(vocab=vocab, dim=dim, var_buckets=var_buckets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0

        for start in range(0, len(statements), batch_size):
            batch = statements[start:start + batch_size]
            batch_losses = []
            for item in batch:
                logits, labels = model.forward_formula(item.formula)
                batch_losses.append(F.cross_entropy(logits, labels))

            loss = torch.stack(batch_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_items += len(batch)

        avg_loss = total_loss / max(1, total_items)
        losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{epochs} | tree codec loss {avg_loss:.4f}")

    if save_path is not None:
        save_tree_codec(model, save_path)

    return model, losses


def save_tree_codec(model: TrainableTreeCodec, path: str | Path) -> Path:
    _require_torch()
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "vocab": model.vocab,
            "dim": model.dim,
            "var_buckets": model.var_buckets,
            "state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path


def load_tree_codec(path: str | Path, device: str | None = None) -> TrainableTreeCodec:
    _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model = TrainableTreeCodec(
        vocab=checkpoint["vocab"],
        dim=int(checkpoint["dim"]),
        var_buckets=int(checkpoint["var_buckets"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model
