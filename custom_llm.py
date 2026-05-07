from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from logic import Formula
from statement_generation import LabeledStatement, formula_to_text


PAD_TOKEN = "<PAD>"
TRUE_TOKEN = "true"
FALSE_TOKEN = "false"

MODE_REGULAR = "regular"
MODE_DEPTH = "depth"
MODE_SIDE = "side"
MODE_LOGICAL = "logical"
MODE_DISTANCE = "distance"
MODE_ALL = "all"

EMBEDDING_MODES = (
    MODE_REGULAR,
    MODE_DEPTH,
    MODE_SIDE,
    MODE_LOGICAL,
    MODE_DISTANCE,
    MODE_ALL,
)

EMBEDDING_MODE_LABELS = {
    MODE_REGULAR: "regular token + position embeddings",
    MODE_DEPTH: "token + position + parenthesis-depth embeddings",
    MODE_SIDE: "token + position + LHS/RHS embeddings",
    MODE_LOGICAL: "token + position + logical-role embeddings",
    MODE_DISTANCE: "token + position + implication-distance embeddings",
    MODE_ALL: "token + position + all structural embeddings",
}

EMBEDDING_MODE_ALIASES = {
    "1": MODE_REGULAR,
    "regular": MODE_REGULAR,
    "base": MODE_REGULAR,
    "2": MODE_DEPTH,
    "depth": MODE_DEPTH,
    "parentheses": MODE_DEPTH,
    "parenthesis": MODE_DEPTH,
    "3": MODE_SIDE,
    "side": MODE_SIDE,
    "lhs_rhs": MODE_SIDE,
    "lhs-vs-rhs": MODE_SIDE,
    "4": MODE_LOGICAL,
    "logical": MODE_LOGICAL,
    "logic": MODE_LOGICAL,
    "5": MODE_DISTANCE,
    "distance": MODE_DISTANCE,
    "implication-distance": MODE_DISTANCE,
    "6": MODE_ALL,
    "all": MODE_ALL,
    "combined": MODE_ALL,
}

LOGICAL_PAD = 0
LOGICAL_VARIABLE = 1
LOGICAL_PARAMETER = 2
LOGICAL_PARENTHESIS = 3
LOGICAL_OR = 4
LOGICAL_IMPLICATION = 5
LOGICAL_AND = 6

SIDE_NONE = 0
SIDE_LHS = 1
SIDE_IMPLICATION = 2
SIDE_RHS = 3


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ModuleNotFoundError("custom_llm requires torch")


def normalize_embedding_mode(mode: str | int | None) -> str:
    if mode is None:
        return MODE_REGULAR
    normalized = str(mode).strip().lower().replace(" ", "_")
    if normalized in EMBEDDING_MODE_ALIASES:
        return EMBEDDING_MODE_ALIASES[normalized]
    valid = ", ".join(EMBEDDING_MODES)
    raise ValueError(f"Unknown embedding mode {mode!r}. Choose one of: {valid}")


@dataclass
class Config:
    d_model: int
    d_hidden: int
    d_head: int
    d_vocab: int
    n_heads: int
    num_blocks: int
    max_seq_len: int = 4096
    embedding_mode: str = MODE_REGULAR
    max_parentheses_depth: int = 64
    max_implication_distance: int = 256

    def __post_init__(self) -> None:
        self.embedding_mode = normalize_embedding_mode(self.embedding_mode)


class WordTokenizer:
    def __init__(self, initial_text: str = "", stoi: dict[str, int] | None = None):
        self.stoi: dict[str, int] = dict(stoi) if stoi is not None else {}
        self.itos: dict[int, str] = {i: w for w, i in self.stoi.items()}
        if stoi is None:
            self.add_word(PAD_TOKEN)
            self.add_word(TRUE_TOKEN)
            self.add_word(FALSE_TOKEN)
        if initial_text:
            self.add_text(initial_text)

    @property
    def pad_id(self) -> int:
        return self.add_word(PAD_TOKEN)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def normalize(self, text: str) -> list[str]:
        tokens = re.findall(r"->|[A-Za-z][A-Za-z0-9_]*|\d+|[()]|[^\s]", text)
        return [token.lower() if token not in {"->", "(", ")"} else token for token in tokens]

    def add_text(self, text: str) -> None:
        for word in self.normalize(text):
            self.add_word(word)

    def add_word(self, word: str) -> int:
        if word in self.stoi:
            return self.stoi[word]
        new_id = len(self.stoi)
        self.stoi[word] = new_id
        self.itos[new_id] = word
        return new_id

    def encode(self, text: str) -> list[int]:
        return [self.add_word(word) for word in self.normalize(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos.get(int(i), PAD_TOKEN) for i in ids)


def make_tokenizer(stoi: dict[str, int] | None = None) -> WordTokenizer:
    return WordTokenizer(stoi=stoi)


def formula_prompt(formula: Formula | str) -> str:
    if isinstance(formula, str):
        return formula.strip()
    return formula_to_text(formula)


def statement_prompt(statement: LabeledStatement) -> str:
    return statement.formula_text


def _logical_type_id(token: str) -> int:
    lower = token.lower()
    if token == PAD_TOKEN:
        return LOGICAL_PAD
    if lower == "or":
        return LOGICAL_OR
    if lower == "and":
        return LOGICAL_AND
    if token == "->":
        return LOGICAL_IMPLICATION
    if token in {"(", ")"}:
        return LOGICAL_PARENTHESIS
    if lower in {TRUE_TOKEN, FALSE_TOKEN} or re.fullmatch(r"\d+", lower):
        return LOGICAL_PARAMETER
    if re.fullmatch(r"[a-z][a-z0-9_]*", lower):
        return LOGICAL_VARIABLE
    return LOGICAL_PARAMETER


def token_feature_ids(
    tokens: Sequence[str],
    *,
    max_parentheses_depth: int = 64,
    max_implication_distance: int = 256,
) -> dict[str, list[int]]:
    depth_ids: list[int] = []
    side_ids: list[int] = []
    logical_ids: list[int] = []
    distance_ids: list[int] = []

    implication_positions = [index for index, token in enumerate(tokens) if token == "->"]
    saw_implication = False
    depth = 0

    for index, token in enumerate(tokens):
        if token == PAD_TOKEN:
            depth_ids.append(0)
            side_ids.append(SIDE_NONE)
            logical_ids.append(LOGICAL_PAD)
            distance_ids.append(0)
            continue

        if token == "(":
            depth += 1
            depth_ids.append(min(depth, max_parentheses_depth))
        elif token == ")":
            depth_ids.append(min(max(depth, 0), max_parentheses_depth))
            depth = max(depth - 1, 0)
        else:
            depth_ids.append(min(depth, max_parentheses_depth))

        if token == "->":
            side_ids.append(SIDE_IMPLICATION)
            saw_implication = True
        elif saw_implication:
            side_ids.append(SIDE_RHS)
        elif implication_positions:
            side_ids.append(SIDE_LHS)
        else:
            side_ids.append(SIDE_NONE)

        logical_ids.append(_logical_type_id(token))

        if implication_positions:
            distance = min(abs(index - position) for position in implication_positions)
            distance_ids.append(min(distance, max_implication_distance))
        else:
            distance_ids.append(max_implication_distance)

    return {
        "depth": depth_ids,
        "side": side_ids,
        "logical": logical_ids,
        "distance": distance_ids,
    }


if nn is None:

    class CustomLogicTransformer:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("CustomLogicTransformer requires torch")

else:

    class Embedding(nn.Module):
        def __init__(self, config: Config, d_vocab: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(d_vocab, config.d_model))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dtype != torch.long:
                x = x.long()
            return self.weight[x]


    class MLP(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.linear1 = nn.Linear(config.d_model, config.d_hidden)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(config.d_hidden, config.d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear2(self.relu(self.linear1(x)))


    class AttentionHead(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.n_heads = config.n_heads
            self.d_head = config.d_head
            self.W_q = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
            self.W_k = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
            self.W_v = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
            self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(0)

            batch, tokens, _ = x.shape
            heads, head_dim = self.n_heads, self.d_head

            q = self.W_q(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
            k = self.W_k(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
            v = self.W_v(x).view(batch, tokens, heads, head_dim).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            if causal_mask is None:
                causal_mask = torch.triu(torch.ones(tokens, tokens, device=x.device, dtype=torch.bool), diagonal=1)

            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).contiguous().view(batch, tokens, heads * head_dim)
            return self.W_o(out)


    class TransformerBlock(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.ln1 = nn.LayerNorm(config.d_model)
            self.attn = AttentionHead(config)
            self.ln2 = nn.LayerNorm(config.d_model)
            self.mlp = MLP(config)

        def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
            x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
            x = x + self.mlp(self.ln2(x))
            return x


    class CustomLogicTransformer(nn.Module):
        def __init__(self, config: Config, tokenizer: WordTokenizer):
            super().__init__()
            self.config = config
            self.tokenizer = tokenizer
            self.embed = Embedding(config, tokenizer.vocab_size)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.depth_emb = nn.Embedding(config.max_parentheses_depth + 1, config.d_model)
            self.side_emb = nn.Embedding(4, config.d_model)
            self.logical_emb = nn.Embedding(7, config.d_model)
            self.distance_emb = nn.Embedding(config.max_implication_distance + 1, config.d_model)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
            self.ln_f = nn.LayerNorm(config.d_model)
            self.unembed = nn.Linear(config.d_model, tokenizer.vocab_size)
            self.register_buffer("_causal_mask", None, persistent=False)
            self._init_std = 0.02

        def _get_causal_mask(self, tokens: int, device: torch.device) -> torch.Tensor:
            if self._causal_mask is None or self._causal_mask.size(0) != tokens or self._causal_mask.device != device:
                self._causal_mask = torch.triu(torch.ones(tokens, tokens, device=device, dtype=torch.bool), diagonal=1)
            return self._causal_mask

        def resize_vocab(self, new_vocab_size: int) -> None:
            old_vocab_size = self.embed.weight.size(0)
            if new_vocab_size <= old_vocab_size:
                return

            device = self.embed.weight.device
            dtype = self.embed.weight.dtype
            d_model = self.embed.weight.size(1)

            new_embed = nn.Parameter(torch.empty(new_vocab_size, d_model, device=device, dtype=dtype))
            nn.init.normal_(new_embed, mean=0.0, std=self._init_std)
            new_embed.data[:old_vocab_size] = self.embed.weight.data
            self.embed.weight = new_embed

            old_unembed = self.unembed
            new_unembed = nn.Linear(d_model, new_vocab_size, bias=True).to(device=device, dtype=dtype)
            nn.init.normal_(new_unembed.weight, mean=0.0, std=self._init_std)
            nn.init.zeros_(new_unembed.bias)
            new_unembed.weight.data[:old_vocab_size] = old_unembed.weight.data
            new_unembed.bias.data[:old_vocab_size] = old_unembed.bias.data
            self.unembed = new_unembed

        def _feature_tensors(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            rows = token_ids.tolist()
            feature_rows = {"depth": [], "side": [], "logical": [], "distance": []}

            for row in rows:
                tokens = [self.tokenizer.itos.get(int(token_id), PAD_TOKEN) for token_id in row]
                features = token_feature_ids(
                    tokens,
                    max_parentheses_depth=self.config.max_parentheses_depth,
                    max_implication_distance=self.config.max_implication_distance,
                )
                for key in feature_rows:
                    feature_rows[key].append(features[key])

            return {
                key: torch.tensor(value, dtype=torch.long, device=token_ids.device)
                for key, value in feature_rows.items()
            }

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            batch, tokens = token_ids.shape
            if tokens > self.config.max_seq_len:
                raise ValueError(f"Sequence length {tokens} exceeds max_seq_len={self.config.max_seq_len}")

            token_embedding = self.embed(token_ids)
            pos_ids = torch.arange(tokens, device=token_ids.device)
            positional_embedding = self.pos_emb(pos_ids).unsqueeze(0).expand(batch, -1, -1)
            x = token_embedding + positional_embedding

            mode = normalize_embedding_mode(self.config.embedding_mode)
            if mode != MODE_REGULAR:
                features = self._feature_tensors(token_ids)
                if mode in {MODE_DEPTH, MODE_ALL}:
                    x = x + self.depth_emb(features["depth"])
                if mode in {MODE_SIDE, MODE_ALL}:
                    x = x + self.side_emb(features["side"])
                if mode in {MODE_LOGICAL, MODE_ALL}:
                    x = x + self.logical_emb(features["logical"])
                if mode in {MODE_DISTANCE, MODE_ALL}:
                    x = x + self.distance_emb(features["distance"])

            mask = self._get_causal_mask(tokens, token_ids.device)
            for block in self.blocks:
                x = block(x, causal_mask=mask)

            x = self.ln_f(x)
            return self.unembed(x)


def make_default_config(embedding_mode: str = MODE_REGULAR) -> Config:
    return Config(
        d_model=256,
        d_hidden=1024,
        d_head=64,
        d_vocab=0,
        n_heads=4,
        num_blocks=6,
        max_seq_len=4096,
        embedding_mode=embedding_mode,
        max_parentheses_depth=64,
        max_implication_distance=256,
    )


def make_custom_logic_llm(
    config: Config | None = None,
    embedding_mode: str | None = None,
) -> CustomLogicTransformer:
    _require_torch()
    if config is None:
        config = make_default_config(embedding_mode=normalize_embedding_mode(embedding_mode))
    elif embedding_mode is not None:
        config.embedding_mode = normalize_embedding_mode(embedding_mode)
    tokenizer = make_tokenizer()
    return CustomLogicTransformer(config, tokenizer)


def _pad_answer_only_batch(
    prompt_token_lists: Sequence[Sequence[int]],
    answer_token_lists: Sequence[Sequence[int]],
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(prompt_token_lists) != len(answer_token_lists):
        raise ValueError("prompt and answer batches must have the same length")

    rows: list[tuple[list[int], int, int]] = []
    for prompt_ids, answer_ids in zip(prompt_token_lists, answer_token_lists):
        prompt = list(prompt_ids)
        answer = list(answer_ids)
        if not prompt:
            raise ValueError("Prompt token list must not be empty")
        if not answer:
            raise ValueError("Answer token list must not be empty")
        if len(answer) != 1:
            raise ValueError(
                "Labeled-statement answers must tokenize to exactly one token "
                "so the model cannot see any part of the label in its input."
            )

        full_ids = prompt + answer
        rows.append((full_ids, len(prompt), len(answer)))

    max_len = max(len(full_ids) - 1 for full_ids, _, _ in rows)
    x = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(rows), max_len), -100, dtype=torch.long, device=device)

    for row, (full_ids, prompt_len, answer_len) in enumerate(rows):
        x_ids = full_ids[:-1]
        y_ids = full_ids[1:]
        length = len(x_ids)
        x[row, :length] = torch.tensor(x_ids, dtype=torch.long, device=device)

        answer_start = prompt_len - 1
        answer_end = answer_start + answer_len
        y[row, answer_start:answer_end] = torch.tensor(
            y_ids[answer_start:answer_end],
            dtype=torch.long,
            device=device,
        )

    return x, y


def _iter_batches(items: Sequence, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def train_on_labeled_statements(
    model: CustomLogicTransformer,
    statements: Sequence[LabeledStatement],
    epochs: int,
    lr: float,
    batch_size: int = 8,
    device: str | None = None,
    grad_clip: float | None = 1.0,
    save_path: str | Path | None = None,
    progress_interval: int = 25,
) -> list[float]:
    _require_torch()
    if not statements:
        raise ValueError("statements must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = [statement_prompt(statement) for statement in statements]
    answers = [TRUE_TOKEN if statement.label else FALSE_TOKEN for statement in statements]

    for prompt, answer in zip(prompts, answers):
        model.tokenizer.add_text(prompt)
        model.tokenizer.add_text(answer)
    pad_id = model.tokenizer.pad_id
    model.resize_vocab(model.tokenizer.vocab_size)

    encoded_samples = [
        (model.tokenizer.encode(prompt), model.tokenizer.encode(answer))
        for prompt, answer in zip(prompts, answers)
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    update_index = 0
    total_updates = epochs * ((len(encoded_samples) + batch_size - 1) // batch_size)

    for epoch_index in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_items = 0

        for batch in _iter_batches(encoded_samples, batch_size):
            prompt_ids = [item[0] for item in batch]
            answer_ids = [item[1] for item in batch]
            input_tokens = max(len(prompt) + len(answer) - 1 for prompt, answer in batch)
            if input_tokens > model.config.max_seq_len:
                print(
                    f"[warn] Skipping batch in epoch {epoch_index}: "
                    f"tokenized length {input_tokens} exceeds max_seq_len={model.config.max_seq_len}"
                )
                continue

            x, y = _pad_answer_only_batch(prompt_ids, answer_ids, pad_id=pad_id, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            batch_loss = loss.item()
            losses.append(batch_loss)
            epoch_loss += batch_loss * len(batch)
            epoch_items += len(batch)
            update_index += 1

            if update_index % progress_interval == 0 or update_index == total_updates:
                print(
                    f"epoch {epoch_index}/{epochs} | update {update_index}/{total_updates} | "
                    f"{model.config.embedding_mode} answer loss {batch_loss:.4f}"
                )

        if epoch_items:
            avg_loss = epoch_loss / epoch_items
            print(f"epoch {epoch_index}/{epochs} | average answer loss {avg_loss:.4f}")

    if save_path is not None:
        save_checkpoint(model, model.config, save_path)

    return losses


# Backward-compatible name for older callers in this repository.
def train_on_labeled_statements_with_ast(
    model: CustomLogicTransformer,
    statements: Sequence[LabeledStatement],
    epochs: int,
    lr: float,
    batch_size: int = 8,
    device: str | None = None,
    grad_clip: float | None = 1.0,
    save_path: str | Path | None = None,
    **_: object,
) -> list[float]:
    return train_on_labeled_statements(
        model=model,
        statements=statements,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        grad_clip=grad_clip,
        save_path=save_path,
    )


@torch.no_grad() if torch is not None else (lambda fn: fn)
def predict_truth_label(
    model: CustomLogicTransformer,
    formula: Formula | str,
    device: str | None = None,
) -> tuple[bool, float]:
    _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    model.tokenizer.add_word(TRUE_TOKEN)
    model.tokenizer.add_word(FALSE_TOKEN)
    prompt = formula_prompt(formula)
    ids = model.tokenizer.encode(prompt)
    model.resize_vocab(model.tokenizer.vocab_size)

    token_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(token_tensor)[0, -1]

    true_id = model.tokenizer.stoi[TRUE_TOKEN]
    false_id = model.tokenizer.stoi[FALSE_TOKEN]
    selected_logits = torch.stack([logits[false_id], logits[true_id]])
    probs = F.softmax(selected_logits, dim=0)
    false_prob = float(probs[0].item())
    true_prob = float(probs[1].item())
    prediction = true_prob >= false_prob
    confidence = true_prob if prediction else false_prob
    return prediction, confidence


@torch.no_grad() if torch is not None else (lambda fn: fn)
def predict_truth_text(
    model: CustomLogicTransformer,
    formula: Formula | str,
    device: str | None = None,
    **_: object,
) -> str:
    prediction, confidence = predict_truth_label(model, formula, device=device)
    return f"{TRUE_TOKEN if prediction else FALSE_TOKEN} ({confidence:.2%})"


def save_checkpoint(model: CustomLogicTransformer, config: Config, path: str | Path) -> Path:
    _require_torch()
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "config": asdict(config),
            "stoi": model.tokenizer.stoi,
            "embedding_mode": config.embedding_mode,
            "model_state": model.state_dict(),
        },
        output_path,
    )
    return output_path


def _config_from_checkpoint(raw_config: dict, embedding_mode: str | None = None) -> Config:
    valid_fields = {field.name for field in fields(Config)}
    cleaned = {key: value for key, value in raw_config.items() if key in valid_fields}
    defaults = asdict(make_default_config())
    defaults.update(cleaned)
    if embedding_mode is not None:
        defaults["embedding_mode"] = embedding_mode
    return Config(**defaults)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[CustomLogicTransformer, Config] | None:
    _require_torch()
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    config = _config_from_checkpoint(
        dict(checkpoint.get("config", {})),
        embedding_mode=checkpoint.get("embedding_mode"),
    )
    tokenizer = make_tokenizer(stoi=checkpoint["stoi"])
    model = CustomLogicTransformer(config, tokenizer).to(device)
    model.resize_vocab(model.tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, config
