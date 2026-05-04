from __future__ import annotations

"""
AST-conditioned adaptation of the user's LLM3.py Transformer.

The original model is a causal next-token Transformer. This version keeps that
shape and adds a tree_embedding input projected into the token stream.
"""

import math
import os
import re
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
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
from statement_generation import LabeledStatement, statement_text_for_formula
from tree_encoding import FormulaTreeEncoder, HashingTreeEmbedder


LOGIC_SPECIAL_TOKENS = [
    "<PAD>",
    "<LPAREN>",
    "<RPAREN>",
    "<AND>",
    "<OR>",
    "<IMP>",
    "<TRUE>",
    "<FALSE>",
    "<STATEMENT>",
    "<QUESTION>",
    "<ANSWER>",
]

PAD_TOKEN = "<PAD>"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {"User-Agent": "LogicTruthASTProject/1.0"}


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ModuleNotFoundError("custom_llm requires torch")


@dataclass
class Config:
    d_model: int
    d_hidden: int
    d_head: int
    d_vocab: int
    n_heads: int
    num_blocks: int
    tree_embedding_dim: int = 256
    tree_prefix_tokens: int = 4
    max_seq_len: int = 4096


class WordTokenizer:
    def __init__(self, initial_text: str = "", stoi: dict[str, int] | None = None):
        self.stoi: dict[str, int] = dict(stoi) if stoi is not None else {}
        self.itos: dict[int, str] = {i: w for w, i in self.stoi.items()}
        if stoi is None:
            self.add_word(PAD_TOKEN)
        if initial_text:
            self.add_text(initial_text)

    @property
    def pad_id(self) -> int:
        return self.add_word(PAD_TOKEN)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def normalize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\w+(?:[-']\w+)*|[.,:\"()!?/\\>-]", text)

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
        ids: list[int] = []
        for word in self.normalize(text):
            ids.append(self.add_word(word))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)


class LogicAwareTokenizer(WordTokenizer):
    """
    Tokenizer for the small custom model.

    It gives logical syntax stable tokens instead of leaving operators split
    across punctuation. Variables become VAR_P, VAR_Q, etc.
    """

    def __init__(self, initial_text: str = "", stoi: dict[str, int] | None = None):
        super().__init__(initial_text="", stoi=stoi)
        if stoi is None:
            for token in LOGIC_SPECIAL_TOKENS:
                self.add_word(token)
        if initial_text:
            self.add_text(initial_text)

    def normalize(self, text: str) -> list[str]:
        text = (
            text.replace("/\\", " <AND> ")
            .replace("\\/", " <OR> ")
            .replace("->", " <IMP> ")
            .replace("(", " <LPAREN> ")
            .replace(")", " <RPAREN> ")
        )

        raw_tokens = re.findall(r"<[A-Z]+>|[A-Za-z][A-Za-z0-9_]*|\d+|[.,:\"!?-]", text)
        tokens: list[str] = []

        for token in raw_tokens:
            if token in LOGIC_SPECIAL_TOKENS:
                tokens.append(token)
                continue

            lower = token.lower()
            if lower == "true":
                tokens.append("<TRUE>")
            elif lower == "false":
                tokens.append("<FALSE>")
            elif lower == "statement":
                tokens.append("<STATEMENT>")
            elif lower == "question":
                tokens.append("<QUESTION>")
            elif lower == "answer":
                tokens.append("<ANSWER>")
            elif re.fullmatch(r"[A-Z][A-Z0-9_]*", token):
                tokens.append(f"VAR_{token}")
            else:
                tokens.append(lower)

        return tokens


def make_tokenizer(tokenizer_kind: str = "word", stoi: dict[str, int] | None = None) -> WordTokenizer:
    tokenizer_kind = tokenizer_kind.lower().strip()
    initial_text = "" if stoi is not None else "statement answer true false"
    if tokenizer_kind == "logic":
        return LogicAwareTokenizer(initial_text, stoi=stoi)
    if tokenizer_kind == "word":
        return WordTokenizer(initial_text, stoi=stoi)
    raise ValueError("tokenizer_kind must be 'word' or 'logic'")


def logic_token_text(text: str) -> str:
    tokenizer = LogicAwareTokenizer()
    return " ".join(tokenizer.normalize(text))


def logic_formula_prompt(formula: Formula) -> str:
    return (
        "<STATEMENT>\n"
        f"{logic_token_text(statement_text_for_formula(formula))}\n"
        "<QUESTION> true under every truth assignment ?\n"
        "<ANSWER>:"
    )


def wikipedia_title_from_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    if line.startswith("http://") or line.startswith("https://"):
        parsed = urllib.parse.urlparse(line)
        if "/wiki/" in parsed.path:
            title = parsed.path.split("/wiki/", 1)[1]
            return urllib.parse.unquote(title).replace("_", " ")

    return line


def fetch_wikipedia_article_text(title_or_url: str, timeout: int = 20) -> str:
    title = wikipedia_title_from_line(title_or_url)
    if not title:
        return ""

    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "format": "json",
            "redirects": "1",
            "titles": title,
        }
    )
    request = urllib.request.Request(
        f"{WIKIPEDIA_API}?{query}",
        headers=WIKIPEDIA_HEADERS,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = (page.get("extract") or "").strip()
        if extract:
            return extract

    return ""


def read_wikipedia_titles_file(path: str | Path) -> list[str]:
    titles: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            title = wikipedia_title_from_line(line)
            if title and not title.startswith("#"):
                titles.append(title)
    return titles


if nn is None:

    class AstConditionedTransformer:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("AstConditionedTransformer requires torch")

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


    class AstConditionedTransformer(nn.Module):
        """
        Your LLM3-style causal Transformer with an additional AST vector input.

        The tree vector is projected into d_model and added to each token
        representation. This keeps the original next-token objective but gives
        the network a parallel structural signal while it reads the statement.
        """

        def __init__(self, config: Config, tokenizer: WordTokenizer):
            super().__init__()
            self.config = config
            self.tokenizer = tokenizer
            self.embed = Embedding(config, tokenizer.vocab_size)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.tree_proj = nn.Linear(
                config.tree_embedding_dim,
                config.tree_prefix_tokens * config.d_model,
                bias=False,
            )
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

        def forward(self, token_ids: torch.Tensor, tree_embedding: torch.Tensor | None = None) -> torch.Tensor:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            batch, tokens = token_ids.shape
            extra_tokens = self.config.tree_prefix_tokens if tree_embedding is not None else 0
            if tokens + extra_tokens > self.config.max_seq_len:
                raise ValueError(f"Sequence length {tokens + extra_tokens} exceeds max_seq_len={self.config.max_seq_len}")

            tok = self.embed(token_ids)
            pos_ids = torch.arange(tokens, device=token_ids.device)
            pos = self.pos_emb(pos_ids).unsqueeze(0)
            x = tok + pos

            if tree_embedding is not None:
                if tree_embedding.dim() == 1:
                    tree_embedding = tree_embedding.unsqueeze(0)
                if tree_embedding.size(0) == 1 and batch > 1:
                    tree_embedding = tree_embedding.expand(batch, -1)
                prefix = self.tree_proj(tree_embedding.to(device=x.device, dtype=x.dtype))
                prefix = prefix.view(batch, self.config.tree_prefix_tokens, self.config.d_model)
                x = torch.cat([prefix, x], dim=1)

            mask = self._get_causal_mask(x.size(1), x.device)
            for block in self.blocks:
                x = block(x, causal_mask=mask)

            x = self.ln_f(x)
            if tree_embedding is not None:
                x = x[:, self.config.tree_prefix_tokens:, :]
            return self.unembed(x)


def make_default_config(tree_embedding_dim: int = 256, tree_prefix_tokens: int = 4) -> Config:
    return Config(
        d_model=256,
        d_hidden=1024,
        d_head=64,
        d_vocab=0,
        n_heads=4,
        num_blocks=6,
        tree_embedding_dim=tree_embedding_dim,
        tree_prefix_tokens=tree_prefix_tokens,
        max_seq_len=4096,
    )


def make_custom_logic_llm(
    config: Config | None = None,
    tokenizer_kind: str = "word",
) -> AstConditionedTransformer:
    _require_torch()
    tokenizer = make_tokenizer(tokenizer_kind)
    tokenizer.tokenizer_kind = tokenizer_kind
    config = config or make_default_config()
    return AstConditionedTransformer(config, tokenizer)


def statement_prompt(statement: LabeledStatement, include_label: bool = True) -> str:
    prompt = (
        "Statement:\n"
        f"{statement.text}\n"
        "Question: Is it true under every truth assignment?\n"
        "Answer:"
    )
    if include_label:
        prompt += f" {'true' if statement.label else 'false'}"
    return prompt


def formula_prompt(formula: Formula) -> str:
    return (
        "Statement:\n"
        f"{statement_text_for_formula(formula)}\n"
        "Question: Is it true under every truth assignment?\n"
        "Answer:"
    )


def _hash_tree_embedding(formula: Formula, dim: int, device: str) -> torch.Tensor:
    embedder = HashingTreeEmbedder(dim=dim)
    return embedder.encode_formula(formula).to(device)


def _trained_tree_embedding(encoder: FormulaTreeEncoder, formula: Formula, device: str) -> torch.Tensor:
    encoder = encoder.to(device)
    return encoder.encode_formula(formula)


def tree_embedding_for_formula(
    formula: Formula,
    dim: int,
    device: str,
    trained_encoder: FormulaTreeEncoder | None = None,
) -> torch.Tensor:
    _require_torch()
    if trained_encoder is not None:
        embedding = _trained_tree_embedding(trained_encoder, formula, device)
    else:
        embedding = _hash_tree_embedding(formula, dim, device)

    if embedding.numel() != dim:
        raise ValueError(f"Tree embedding has dim {embedding.numel()}, but model expects {dim}")

    return embedding


def _pad_next_token_batch(
    token_lists: Sequence[Sequence[int]],
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    usable = [list(ids) for ids in token_lists if len(ids) >= 2]
    if not usable:
        raise ValueError("Batch has no sequences with at least two tokens")

    max_len = max(len(ids) - 1 for ids in usable)
    x = torch.full((len(usable), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(usable), max_len), -100, dtype=torch.long, device=device)

    for row, ids in enumerate(usable):
        x_ids = ids[:-1]
        y_ids = ids[1:]
        length = len(x_ids)
        x[row, :length] = torch.tensor(x_ids, dtype=torch.long, device=device)
        y[row, :length] = torch.tensor(y_ids, dtype=torch.long, device=device)

    return x, y


def _iter_batches(items: Sequence, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def top_k_filter(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    vocab = logits.size(-1)
    k = min(k, vocab)
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    cutoff = topk_vals[-1].unsqueeze(-1)
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def generate_sample(
    model: AstConditionedTransformer,
    prompt_tokens: torch.Tensor,
    tree_embedding: torch.Tensor | None = None,
    max_new_tokens: int = 6,
    temperature: float = 0.7,
    top_k: int | None = 20,
) -> torch.Tensor:
    _require_torch()
    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        logits = model(tokens, tree_embedding=tree_embedding)
        next_logits = logits[0, -1] / max(float(temperature), 1e-8)
        next_logits = top_k_filter(next_logits, top_k)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=0)

    return tokens


def train_on_labeled_statements_with_ast(
    model: AstConditionedTransformer,
    statements: Sequence[LabeledStatement],
    epochs: int,
    lr: float,
    batch_size: int = 8,
    device: str | None = None,
    grad_clip: float | None = 1.0,
    trained_encoder: FormulaTreeEncoder | None = None,
    train_tree_encoder: bool = False,
    use_ast: bool = True,
    save_path: str | Path | None = None,
) -> list[float]:
    _require_torch()
    if not statements:
        raise ValueError("statements must not be empty")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if trained_encoder is not None:
        trained_encoder.to(device)
        trained_encoder.train(mode=train_tree_encoder)
        for param in trained_encoder.parameters():
            param.requires_grad = train_tree_encoder

    texts = [statement_prompt(statement, include_label=True) for statement in statements]
    for text in texts:
        model.tokenizer.add_text(text)
    pad_id = model.tokenizer.pad_id
    model.resize_vocab(model.tokenizer.vocab_size)

    params = list(model.parameters())
    if use_ast and trained_encoder is not None and train_tree_encoder:
        params.extend(p for p in trained_encoder.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=lr)
    losses: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0

        for batch in _iter_batches(list(zip(statements, texts)), batch_size):
            batch_statements = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            encoded = [model.tokenizer.encode(text) for text in batch_texts]
            x, y = _pad_next_token_batch(encoded, pad_id=pad_id, device=device)

            tree_vec = None
            if use_ast:
                tree_vec = torch.stack(
                    [
                        tree_embedding_for_formula(
                            statement.formula,
                            dim=model.config.tree_embedding_dim,
                            device=device,
                            trained_encoder=trained_encoder,
                        )
                        for statement in batch_statements
                    ]
                )

            logits = model(x, tree_embedding=tree_vec)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            optimizer.step()
            total_loss += loss.item() * len(batch_statements)
            total_items += len(batch_statements)

        avg_loss = total_loss / max(1, total_items)
        losses.append(avg_loss)
        label = "AST-conditioned LLM" if use_ast else "text-only custom LLM"
        print(f"epoch {epoch + 1}/{epochs} | {label} loss {avg_loss:.4f}")

    if save_path is not None:
        save_checkpoint(model, model.config, save_path)

    return losses


def _make_token_chunks(token_ids: Sequence[int], seq_len: int) -> list[list[int]]:
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")

    window = seq_len + 1
    chunks: list[list[int]] = []
    for start in range(0, max(0, len(token_ids) - 1), seq_len):
        chunk = list(token_ids[start:start + window])
        if len(chunk) >= 2:
            chunks.append(chunk)
    return chunks


def train_custom_llm_on_texts(
    model: AstConditionedTransformer,
    texts: Sequence[str],
    epochs: int,
    lr: float,
    batch_size: int = 8,
    seq_len: int = 128,
    device: str | None = None,
    grad_clip: float | None = 1.0,
    save_path: str | Path | None = None,
) -> list[float]:
    """
    Train the whole custom causal LM, including regular token embeddings.

    This is text-only next-token training. It intentionally does not use AST
    soft-prefix embeddings because Wikipedia prose is not a Formula AST.
    """
    _require_torch()
    clean_texts = [text.strip() for text in texts if text and text.strip()]
    if not clean_texts:
        raise ValueError("texts must contain at least one non-empty string")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for text in clean_texts:
        model.tokenizer.add_text(text)
    pad_id = model.tokenizer.pad_id
    model.resize_vocab(model.tokenizer.vocab_size)

    chunks: list[list[int]] = []
    for text in clean_texts:
        chunks.extend(_make_token_chunks(model.tokenizer.encode(text), seq_len=seq_len))

    if not chunks:
        raise ValueError("No trainable token chunks were produced from the text")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0

        for batch in _iter_batches(chunks, batch_size):
            x, y = _pad_next_token_batch(batch, pad_id=pad_id, device=device)
            logits = model(x, tree_embedding=None)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_items += len(batch)

        avg_loss = total_loss / max(1, total_items)
        losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{epochs} | custom text LM loss {avg_loss:.4f}")

    if save_path is not None:
        save_checkpoint(model, model.config, save_path)

    return losses


def train_custom_llm_on_text_file(
    model: AstConditionedTransformer,
    text_path: str | Path,
    epochs: int,
    lr: float,
    batch_size: int = 8,
    seq_len: int = 128,
    device: str | None = None,
    grad_clip: float | None = 1.0,
    save_path: str | Path | None = None,
) -> list[float]:
    text = Path(text_path).read_text(encoding="utf-8")
    return train_custom_llm_on_texts(
        model=model,
        texts=[text],
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        grad_clip=grad_clip,
        save_path=save_path,
    )


def train_custom_llm_on_wikipedia_file(
    model: AstConditionedTransformer,
    titles_path: str | Path,
    epochs: int,
    lr: float,
    batch_size: int = 8,
    seq_len: int = 128,
    device: str | None = None,
    grad_clip: float | None = 1.0,
    save_path: str | Path | None = None,
    min_chars: int = 200,
) -> list[float]:
    titles = read_wikipedia_titles_file(titles_path)
    if not titles:
        raise ValueError(f"No Wikipedia titles found in {titles_path}")

    texts: list[str] = []
    for index, title in enumerate(titles, start=1):
        print(f"[wiki {index}/{len(titles)}] fetching {title}")
        try:
            text = fetch_wikipedia_article_text(title)
        except Exception as exc:
            print(f"[warn] skipping {title}: {exc}")
            continue

        if len(text) < min_chars:
            print(f"[warn] skipping {title}: article text too short")
            continue

        texts.append(f"Title: {title}\n{text}")

    if not texts:
        raise ValueError("No Wikipedia articles were fetched successfully")

    return train_custom_llm_on_texts(
        model=model,
        texts=texts,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        grad_clip=grad_clip,
        save_path=save_path,
    )


@torch.no_grad() if torch is not None else (lambda fn: fn)
def predict_truth_text(
    model: AstConditionedTransformer,
    formula: Formula,
    device: str | None = None,
    trained_encoder: FormulaTreeEncoder | None = None,
    use_ast: bool = True,
) -> str:
    _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt = logic_formula_prompt(formula) if getattr(model.tokenizer, "tokenizer_kind", "word") == "logic" else formula_prompt(formula)
    ids = model.tokenizer.encode(prompt)
    model.resize_vocab(model.tokenizer.vocab_size)
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    tree_vec = None
    if use_ast:
        tree_vec = tree_embedding_for_formula(
            formula,
            dim=model.config.tree_embedding_dim,
            device=device,
            trained_encoder=trained_encoder,
        )
    out = generate_sample(model, tokens, tree_embedding=tree_vec)
    generated_ids = out[len(ids):]
    return model.tokenizer.decode(generated_ids.tolist())


def save_checkpoint(model: AstConditionedTransformer, config: Config, path: str | Path) -> Path:
    _require_torch()
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "config": {
                "d_model": config.d_model,
                "d_hidden": config.d_hidden,
                "d_head": config.d_head,
                "d_vocab": 0,
                "n_heads": config.n_heads,
                "num_blocks": config.num_blocks,
                "tree_embedding_dim": config.tree_embedding_dim,
                "tree_prefix_tokens": config.tree_prefix_tokens,
                "max_seq_len": config.max_seq_len,
            },
            "stoi": model.tokenizer.stoi,
            "tokenizer_kind": getattr(model.tokenizer, "tokenizer_kind", "word"),
            "model_state": model.state_dict(),
        },
        output_path,
    )
    return output_path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[AstConditionedTransformer, Config] | None:
    _require_torch()
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    config_dict = dict(checkpoint["config"])
    config_dict.setdefault("tree_prefix_tokens", 4)
    config_dict.setdefault("max_seq_len", 4096)
    config = Config(**config_dict)
    tokenizer_kind = checkpoint.get("tokenizer_kind", "word")
    tokenizer = make_tokenizer(tokenizer_kind, stoi=checkpoint["stoi"])
    tokenizer.tokenizer_kind = tokenizer_kind
    model = AstConditionedTransformer(config, tokenizer).to(device)
    model.resize_vocab(model.tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, config
