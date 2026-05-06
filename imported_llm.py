from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from custom_llm import LOGIC_SPECIAL_TOKENS, formula_prompt, logic_formula_prompt, statement_prompt, tree_embedding_for_formula
from statement_generation import LabeledStatement
from tree_encoding import FormulaTreeEncoder


def _require_deps() -> None:
    if torch is None or nn is None or F is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ModuleNotFoundError("imported_llm requires torch and transformers")


if nn is None:

    class ImportedAstLLM:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("ImportedAstLLM requires torch and transformers")

else:

    class ImportedAstLLM(nn.Module):
        """
        HuggingFace causal LM conditioned by learned AST soft-prefix tokens.

        The projected tree vector is prepended to the input embeddings as virtual
        tokens, so the imported model's attention heads can attend to structure.
        """

        def __init__(
            self,
            model_name: str = "microsoft/Phi-3-mini-4k-instruct",
            tree_embedding_dim: int = 256,
            tree_prefix_tokens: int = 4,
            tokenizer_mode: str = "pretrained",
            device: str | None = None,
        ):
            super().__init__()
            _require_deps()
            self.model_name = model_name
            self.tree_embedding_dim = tree_embedding_dim
            self.tree_prefix_tokens = tree_prefix_tokens
            self.tokenizer_mode = tokenizer_mode
            self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.float16 if self.device_name == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            self.model.to(self.device_name)

            if tokenizer_mode == "augmented_logic":
                logic_tokens = LOGIC_SPECIAL_TOKENS + [f"VAR_{chr(code)}" for code in range(ord("A"), ord("Z") + 1)]
                added = self.tokenizer.add_special_tokens({"additional_special_tokens": logic_tokens})
                if added:
                    self.model.resize_token_embeddings(len(self.tokenizer))
            elif tokenizer_mode != "pretrained":
                raise ValueError("tokenizer_mode must be 'pretrained' or 'augmented_logic'")

            hidden_size = self.model.get_input_embeddings().weight.size(1)
            self.tree_projector = nn.Linear(
                tree_embedding_dim,
                tree_prefix_tokens * hidden_size,
                bias=False,
            ).to(self.device_name)

        @property
        def device(self) -> torch.device:
            return next(self.tree_projector.parameters()).device

        def set_train_mode(self, mode: str) -> None:
            mode = mode.lower().strip()
            valid = {"adapter_only", "adapter_and_embeddings", "adapter_and_unembedding", "full"}
            if mode not in valid:
                raise ValueError(f"mode must be one of {sorted(valid)}")

            for param in self.model.parameters():
                param.requires_grad = mode == "full"

            for param in self.tree_projector.parameters():
                param.requires_grad = True

            if mode == "adapter_and_embeddings":
                for param in self.model.get_input_embeddings().parameters():
                    param.requires_grad = True

            if mode == "adapter_and_unembedding":
                output_embeddings = self.model.get_output_embeddings()
                if output_embeddings is not None:
                    for param in output_embeddings.parameters():
                        param.requires_grad = True

        def _prefix(self, tree_embedding: torch.Tensor, batch_size: int) -> torch.Tensor:
            if tree_embedding.dim() == 1:
                tree_embedding = tree_embedding.unsqueeze(0)
            if tree_embedding.size(0) == 1 and batch_size > 1:
                tree_embedding = tree_embedding.expand(batch_size, -1)

            prefix = self.tree_projector(tree_embedding.to(device=self.device, dtype=self.tree_projector.weight.dtype))
            hidden_size = self.model.get_input_embeddings().weight.size(1)
            return prefix.view(batch_size, self.tree_prefix_tokens, hidden_size)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tree_embedding: torch.Tensor | None,
            labels: torch.Tensor | None = None,
        ):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            batch_size = input_ids.size(0)

            token_embeddings = self.model.get_input_embeddings()(input_ids)
            if tree_embedding is not None:
                prefix = self._prefix(tree_embedding, batch_size=batch_size)
                inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
                prefix_mask = torch.ones(batch_size, self.tree_prefix_tokens, dtype=attention_mask.dtype, device=self.device)
                full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                inputs_embeds = token_embeddings
                full_attention_mask = attention_mask

            full_labels = None
            if labels is not None:
                labels = labels.to(self.device)
                if tree_embedding is not None:
                    ignore = torch.full(
                        (batch_size, self.tree_prefix_tokens),
                        -100,
                        dtype=labels.dtype,
                        device=self.device,
                    )
                    full_labels = torch.cat([ignore, labels], dim=1)
                else:
                    full_labels = labels

            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                return_dict=True,
            )


def _training_tensors(tokenizer, prompt: str, answer: str, device: torch.device):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    labels = full_ids.clone()
    labels[:, :prompt_ids.size(1)] = -100
    attention_mask = torch.ones_like(full_ids, device=device)
    return full_ids, attention_mask, labels


def _training_batch_tensors(tokenizer, prompts: Sequence[str], answers: Sequence[str], device: torch.device):
    full_texts = [prompt + answer for prompt, answer in zip(prompts, answers)]
    prompt_lengths = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.size(1)
        for prompt in prompts
    ]
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    for row, prompt_length in enumerate(prompt_lengths):
        labels[row, :prompt_length] = -100

    return input_ids, attention_mask, labels


def train_imported_ast_llm(
    statements: Sequence[LabeledStatement],
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    tree_embedding_dim: int = 256,
    tree_prefix_tokens: int = 4,
    model: ImportedAstLLM | None = None,
    tokenizer_mode: str = "pretrained",
    train_mode: str = "adapter_only",
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_clip: float | None = 1.0,
    trained_encoder: FormulaTreeEncoder | None = None,
    use_ast: bool = True,
    train_tree_encoder: bool = False,
    save_path: str | Path | None = None,
    save_base_weights: bool = False,
) -> tuple[ImportedAstLLM, list[float]]:
    _require_deps()
    if not statements:
        raise ValueError("statements must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not use_ast and train_mode == "adapter_only":
        raise ValueError("adapter_only has no trainable path when AST conditioning is disabled")

    if model is None:
        model = ImportedAstLLM(
            model_name=model_name,
            tree_embedding_dim=tree_embedding_dim,
            tree_prefix_tokens=tree_prefix_tokens,
            tokenizer_mode=tokenizer_mode,
        )
    model.set_train_mode(train_mode)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if trained_encoder is not None:
        trained_encoder.to(model.device)
        trained_encoder.train(mode=train_tree_encoder)
        for param in trained_encoder.parameters():
            param.requires_grad = train_tree_encoder
        if train_tree_encoder:
            params.extend(p for p in trained_encoder.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=lr)
    losses: list[float] = []
    total_updates = len(statements) * epochs
    update_index = 0
    skipped_updates = 0
    last_line_index = 0
    last_epoch_index = 0
    last_loss = 0.0

    for line_index, statement in enumerate(statements, start=1):
        prompt = (
            logic_formula_prompt(statement.formula)
            if model.tokenizer_mode == "augmented_logic"
            else statement_prompt(statement, include_label=False)
        )
        answer = f" {'true' if statement.label else 'false'}"

        for epoch_index in range(1, epochs + 1):
            try:
                input_ids, attention_mask, labels = _training_batch_tensors(
                    model.tokenizer,
                    [prompt],
                    [answer],
                    model.device,
                )
                tree_vec = None
                if use_ast:
                    tree_vec = tree_embedding_for_formula(
                        statement.formula,
                        dim=tree_embedding_dim,
                        device=str(model.device),
                        trained_encoder=trained_encoder,
                    ).unsqueeze(0)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tree_embedding=tree_vec,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)

                optimizer.step()
                update_index += 1
                last_line_index = line_index
                last_epoch_index = epoch_index
                last_loss = loss.item()
                losses.append(last_loss)

                should_print = update_index % 50 == 0 or update_index == total_updates
                if should_print:
                    print(
                        f"line {line_index}/{len(statements)} | epoch {epoch_index}/{epochs} | "
                        f"update {update_index}/{total_updates} | imported AST LLM loss {last_loss:.4f}"
                    )
            except (ValueError, RuntimeError) as exc:
                skipped_updates += 1
                print(
                    f"[warn] Skipping line {line_index}/{len(statements)} epoch {epoch_index}/{epochs}: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

    if skipped_updates:
        print(f"[warn] Skipped {skipped_updates} failed imported-model line-epochs.")
    if not losses:
        print("[warn] No imported statement training updates completed.")
    elif update_index % 50 != 0 and update_index != total_updates:
        print(
            f"line {last_line_index}/{len(statements)} | epoch {last_epoch_index}/{epochs} | "
            f"completed updates {update_index}/{total_updates} | imported AST LLM loss {last_loss:.4f}"
        )

    if save_path is not None:
        save_imported_ast_checkpoint(model, save_path, save_base_weights=save_base_weights)

    return model, losses


@torch.no_grad() if torch is not None else (lambda fn: fn)
def predict_imported_truth_text(
    model: ImportedAstLLM,
    formula,
    trained_encoder: FormulaTreeEncoder | None = None,
    use_ast: bool = True,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
) -> str:
    _require_deps()
    model.eval()
    prompt = logic_formula_prompt(formula) if model.tokenizer_mode == "augmented_logic" else formula_prompt(formula)
    encoded = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    tree_vec = None
    if use_ast:
        tree_vec = tree_embedding_for_formula(
            formula,
            dim=model.tree_embedding_dim,
            device=str(model.device),
            trained_encoder=trained_encoder,
        )

    generated: list[int] = []
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tree_embedding=tree_vec,
        )
        logits = outputs.logits[0, -1]
        if temperature and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(torch.argmax(logits, dim=-1).item())

        generated.append(next_id)
        next_tensor = torch.tensor([[next_id]], dtype=input_ids.dtype, device=model.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tensor, device=model.device)], dim=1)

    return model.tokenizer.decode(generated, skip_special_tokens=True).strip()


def save_imported_ast_checkpoint(
    model: ImportedAstLLM,
    path: str | Path,
    save_base_weights: bool = False,
) -> Path:
    _require_deps()
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": model.model_name,
        "tree_embedding_dim": model.tree_embedding_dim,
        "tree_prefix_tokens": model.tree_prefix_tokens,
        "tokenizer_mode": model.tokenizer_mode,
        "tree_projector": model.tree_projector.state_dict(),
        "save_base_weights": save_base_weights,
    }
    if save_base_weights:
        checkpoint["base_model_state"] = model.model.state_dict()

    torch.save(checkpoint, output_path)
    return output_path


def load_imported_ast_checkpoint(path: str | Path, device: str | None = None) -> ImportedAstLLM:
    _require_deps()
    checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ImportedAstLLM(
        model_name=checkpoint["model_name"],
        tree_embedding_dim=int(checkpoint["tree_embedding_dim"]),
        tree_prefix_tokens=int(checkpoint["tree_prefix_tokens"]),
        tokenizer_mode=checkpoint.get("tokenizer_mode", "pretrained"),
        device=device,
    )
    model.tree_projector.load_state_dict(checkpoint["tree_projector"], strict=True)
    if checkpoint.get("base_model_state") is not None:
        model.model.load_state_dict(checkpoint["base_model_state"], strict=False)
    model.eval()
    return model
