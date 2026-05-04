from __future__ import annotations

from pathlib import Path

from custom_llm import (
    make_custom_logic_llm,
    load_checkpoint as load_custom_llm_checkpoint,
    make_default_config,
    predict_truth_text,
    train_custom_llm_on_text_file,
    train_custom_llm_on_wikipedia_file,
    train_on_labeled_statements_with_ast,
)
from dataset import make_dataset
from search import bfs_proof_search, llm_topk_bfs
from statement_generation import (
    check_statement_truth,
    generate_and_save_labeled_statements,
    load_labeled_statements,
)
from tree_codec import load_tree_codec, train_tree_codec
from tree_encoding import FormulaTreeEncoder, proof_state_to_tree_text
from truth_model import (
    evaluate_truth_embedding_model,
    load_truth_embedding_model,
    train_truth_embedding_model,
    verify_symbolic_labels,
)


WEIGHTS_DIR = Path("weights")
DATA_DIR = Path("data")
CUSTOM_WEIGHTS_DEFAULT = WEIGHTS_DIR / "custom_logic_llm.pt"
CUSTOM_TEXT_WEIGHTS_DEFAULT = WEIGHTS_DIR / "custom_logic_llm_text.pt"
IMPORTED_WEIGHTS_DEFAULT = WEIGHTS_DIR / "imported_ast_adapter.pt"
TRUTH_ENCODER_DEFAULT = WEIGHTS_DIR / "truth_embedding.pt"
TREE_CODEC_DEFAULT = WEIGHTS_DIR / "tree_codec.pt"
GENERATED_DATA_DEFAULT = DATA_DIR / "generated_truth_eval.jsonl"


def ask_int(prompt: str, default: int | None = None, minimum: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}> ").strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
            if minimum is not None and value < minimum:
                print(f"[warn] Enter an integer >= {minimum}.")
                continue
            return value
        except ValueError:
            print("[warn] Enter an integer.")


def ask_float(prompt: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    while True:
        raw = input(f"{prompt} [{default}]> ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if minimum is not None and value < minimum:
                print(f"[warn] Enter a value >= {minimum}.")
                continue
            if maximum is not None and value > maximum:
                print(f"[warn] Enter a value <= {maximum}.")
                continue
            return value
        except ValueError:
            print("[warn] Enter a number.")


def ask_path(prompt: str, default: str | None = None) -> Path:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{prompt}{suffix}> ").strip()
        if not raw and default:
            raw = default
        if raw:
            return Path(raw)
        print("[warn] Enter a filename.")


def ask_optional_path(prompt: str, default: Path | str | None = None) -> Path | None:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}> ").strip()
    if not raw:
        return Path(default) if default else None
    if raw.lower() in {"none", "new", "blank"}:
        return None
    return Path(raw)


def choose_theorem():
    examples = make_dataset()

    print("=" * 60)
    print("Available theorems")
    print("=" * 60)

    for i, (name, state) in enumerate(examples):
        print(f"{i}: {name}")
        print(state)
        print()

    while True:
        idx = ask_int("Choose theorem number", minimum=0)
        if 0 <= idx < len(examples):
            return examples[idx]
        print("Invalid choice. Try again.")


def choose_top_k():
    return ask_int("Choose top-k for LLM search", default=2, minimum=1)


def print_trace(result):
    print("\n" + "=" * 60)
    print("LLM DECISION TRACE")
    print("=" * 60)

    if not result.trace:
        print("No LLM trace recorded.")
        return

    for i, step in enumerate(result.trace, start=1):
        print(f"\nDecision {i}")
        print("-" * 60)
        print(f"Depth: {step.depth}")
        print("State before decision:")
        print(step.state_before)
        print("\nLLM ranked tactics:")
        print(" -> ".join(step.llm_ranked_tactics))
        print("\nTop-k tactics tried:")
        print(" -> ".join(step.tried_tactics))

        if step.valid_tactics_taken:
            print("\nValid branches added:")
            print(" -> ".join(step.valid_tactics_taken))
        else:
            print("\nValid branches added: none")


def print_tree_encodings(state):
    print("\n" + "=" * 60)
    print("AST TREE ENCODINGS")
    print("=" * 60)
    print(proof_state_to_tree_text(state))


def generate_statement_file_menu() -> None:
    print("\n--- Generate true/false statements ---")
    n = ask_int("How many statements should I generate?", default=100, minimum=1)
    output_path = ask_path("Output JSONL filename", default=str(GENERATED_DATA_DEFAULT))
    max_depth = ask_int("Max formula depth", default=3, minimum=0)
    seed_raw = input("Seed (blank for random)> ").strip()
    seed = int(seed_raw) if seed_raw else None
    true_fraction = ask_float("Fraction labeled true", default=0.5, minimum=0.0, maximum=1.0)

    statements = generate_and_save_labeled_statements(
        n=n,
        output_path=output_path,
        max_depth=max_depth,
        seed=seed,
        true_fraction=true_fraction,
    )

    true_count = sum(1 for item in statements if item.label)
    false_count = len(statements) - true_count
    print(f"[ok] Wrote {len(statements)} statements to {output_path}")
    print(f"     true={true_count}, false={false_count}")


def check_statement_file_menu() -> None:
    print("\n--- Check saved statement file ---")
    input_path = ask_path("Input JSONL filename", default=str(GENERATED_DATA_DEFAULT))
    statements = load_labeled_statements(input_path)
    correct, total = verify_symbolic_labels(statements)

    print(f"[ok] Loaded {total} statements from {input_path}")
    print(f"Symbolic labels verified: {correct}/{total}")

    preview_count = min(5, total)
    if preview_count:
        print("\nPreview:")
        for item in statements[:preview_count]:
            checked = check_statement_truth(item.formula)
            print(f"- {item.name}: label={item.label}, checked={checked}, statement={item.formula}")


def train_tree_embedding_menu() -> None:
    print("\n--- Train AST embedding truth model ---")
    input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
    save_path = ask_path("Save trained embedding model as", default=str(TRUTH_ENCODER_DEFAULT))
    statements = load_labeled_statements(input_path)

    dim = ask_int("Embedding dimension", default=128, minimum=8)
    epochs = ask_int("Epochs", default=20, minimum=1)
    batch_size = ask_int("Batch size", default=16, minimum=1)
    lr = ask_float("Learning rate", default=0.001, minimum=0.0)

    try:
        model, _ = train_truth_embedding_model(
            statements=statements,
            dim=dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=save_path,
        )
        metrics = evaluate_truth_embedding_model(model, statements)
        print(f"[ok] Saved trained embedding model to {save_path}")
        print(f"Training-file accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.2%})")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")


def train_tree_codec_menu() -> None:
    print("\n--- Train AST embedding/unembedding codec ---")
    input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
    save_path = ask_path("Save tree codec as", default=str(TREE_CODEC_DEFAULT))
    statements = load_labeled_statements(input_path)

    dim = ask_int("Embedding dimension", default=128, minimum=8)
    epochs = ask_int("Epochs", default=20, minimum=1)
    batch_size = ask_int("Batch size", default=8, minimum=1)
    lr = ask_float("Learning rate", default=0.001, minimum=0.0)

    try:
        train_tree_codec(
            statements=statements,
            dim=dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=save_path,
        )
        print(f"[ok] Saved trained tree codec to {save_path}")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")


def evaluate_tree_embedding_menu() -> None:
    print("\n--- Use trained AST embedding truth model ---")
    model_path = ask_path("Trained embedding model filename", default=str(TRUTH_ENCODER_DEFAULT))
    input_path = ask_path("Evaluation JSONL filename", default=str(GENERATED_DATA_DEFAULT))

    try:
        model = load_truth_embedding_model(model_path)
        statements = load_labeled_statements(input_path)
        metrics = evaluate_truth_embedding_model(model, statements)
        print(f"[ok] Evaluated {metrics['total']} statements")
        print(f"Accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.2%})")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")


def choose_tree_embedding_source():
    print("\nTree embedding source")
    print("0) No AST embedding")
    print("1) Fixed hash tree embedding")
    print("2) Untrained recursive tree encoder")
    print("3) Trained truth-classifier encoder")
    print("4) Trained embedding/unembedding codec encoder")
    choice = input("select [1]> ").strip().lower() or "1"

    if choice == "0":
        return None, 256, "none", False

    if choice == "1":
        return None, 256, "hash", True

    if choice == "2":
        dim = ask_int("Untrained encoder dimension", default=128, minimum=8)
        encoder = FormulaTreeEncoder(dim=dim)
        return encoder, dim, "untrained-recursive", True

    if choice == "3":
        model_path = ask_path("Trained embedding model filename", default=str(TRUTH_ENCODER_DEFAULT))
        model = load_truth_embedding_model(model_path)
        return model.encoder, model.dim, "truth-encoder", True

    if choice == "4":
        codec_path = ask_path("Trained tree codec filename", default=str(TREE_CODEC_DEFAULT))
        codec = load_tree_codec(codec_path)
        return codec.encoder, codec.dim, "tree-codec", True

    print("[warn] Unknown choice. Using hash tree embedding.")
    return None, 256, "hash", True



def custom_llm_menu() -> None:
    print("\n--- Custom LLM3-style model ---")
    try:
        trained_encoder, tree_dim, source_name, use_ast = choose_tree_embedding_source()
        prefix_tokens = ask_int("Virtual AST prefix tokens", default=4, minimum=1) if use_ast else 1
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return

    print("\nCustom tokenizer")
    print("1) Simple word tokenizer")
    print("2) Logic-aware tokenizer")
    tokenizer_choice = input("select [2]> ").strip() or "2"
    tokenizer_kind = "logic" if tokenizer_choice == "2" else "word"

    load_path = ask_optional_path("Custom weights to load (type 'new' for fresh model)", default=CUSTOM_WEIGHTS_DEFAULT)

    try:
        if load_path is not None:
            loaded = load_custom_llm_checkpoint(load_path)
            if loaded is None:
                print(f"[warn] No checkpoint found at {load_path}. Starting fresh.")
                model = make_custom_logic_llm(
                    make_default_config(tree_embedding_dim=tree_dim, tree_prefix_tokens=prefix_tokens),
                    tokenizer_kind=tokenizer_kind,
                )
            else:
                model, _ = loaded
                print(f"[ok] Loaded {load_path}")
                tokenizer_kind = getattr(model.tokenizer, "tokenizer_kind", "word")
        else:
            model = make_custom_logic_llm(
                make_default_config(tree_embedding_dim=tree_dim, tree_prefix_tokens=prefix_tokens),
                tokenizer_kind=tokenizer_kind,
            )

        while True:
            print("\nCustom LLM menu")
            print(f"Embedding source: {source_name}")
            print(f"Tokenizer: {tokenizer_kind}")
            print("1) Tune custom model with selected embedding setting")
            print("2) Run on labeled statement file")
            print("3) Train custom model text-only, no AST baseline")
            print("4) Train custom model on a plain text file")
            print("5) Train custom model on Wikipedia articles from a title/URL file")
            print("x) Back")
            choice = input("select> ").strip().lower()

            if choice in {"x", "back", "q", "quit"}:
                break

            if choice == "1":
                input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
                save_path = ask_path("Save custom weights as", default=str(CUSTOM_WEIGHTS_DEFAULT))
                statements = load_labeled_statements(input_path)
                epochs = ask_int("Epochs", default=5, minimum=1)
                batch_size = ask_int("Batch size", default=8, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                train_encoder = False
                if trained_encoder is not None and use_ast:
                    train_encoder = input("Also train the selected AST encoder during LLM tuning? (y/N)> ").strip().lower() in {"y", "yes"}
                train_on_labeled_statements_with_ast(
                    model=model,
                    statements=statements,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    trained_encoder=trained_encoder,
                    train_tree_encoder=train_encoder,
                    use_ast=use_ast,
                    save_path=save_path,
                )
                label = "AST-conditioned" if use_ast else "no-AST"
                print(f"[ok] Saved {label} custom LLM checkpoint to {save_path}")
                continue

            if choice == "2":
                input_path = ask_path("Evaluation JSONL filename", default=str(GENERATED_DATA_DEFAULT))
                statements = load_labeled_statements(input_path)
                limit = ask_int("How many examples to run?", default=min(10, len(statements)), minimum=1)

                for item in statements[:limit]:
                    generated = predict_truth_text(
                        model=model,
                        formula=item.formula,
                        trained_encoder=trained_encoder,
                        use_ast=use_ast,
                    )
                    print(f"\n{item.name}")
                    print(f"Statement: {item.formula}")
                    print(f"Expected: {'true' if item.label else 'false'}")
                    print(f"LLM says: {generated}")
                continue

            if choice == "3":
                input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
                save_path = ask_path("Save custom weights as", default=str(WEIGHTS_DIR / "custom_logic_llm_text_only.pt"))
                statements = load_labeled_statements(input_path)
                epochs = ask_int("Epochs", default=5, minimum=1)
                batch_size = ask_int("Batch size", default=8, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                train_on_labeled_statements_with_ast(
                    model=model,
                    statements=statements,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    trained_encoder=None,
                    use_ast=False,
                    save_path=save_path,
                )
                print(f"[ok] Saved text-only custom LLM checkpoint to {save_path}")
                continue

            if choice == "4":
                text_path = ask_path("Plain text training file")
                save_path = ask_path("Save custom text weights as", default=str(CUSTOM_TEXT_WEIGHTS_DEFAULT))
                epochs = ask_int("Epochs", default=5, minimum=1)
                batch_size = ask_int("Batch size", default=8, minimum=1)
                seq_len = ask_int("Sequence length", default=128, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                train_custom_llm_on_text_file(
                    model=model,
                    text_path=text_path,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    save_path=save_path,
                )
                print(f"[ok] Saved custom text-trained weights to {save_path}")
                continue

            if choice == "5":
                titles_path = ask_path("Wikipedia titles/URLs file")
                save_path = ask_path("Save custom Wikipedia-trained weights as", default=str(WEIGHTS_DIR / "custom_logic_llm_wikipedia.pt"))
                epochs = ask_int("Epochs", default=3, minimum=1)
                batch_size = ask_int("Batch size", default=8, minimum=1)
                seq_len = ask_int("Sequence length", default=128, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                min_chars = ask_int("Minimum article characters", default=200, minimum=1)
                train_custom_llm_on_wikipedia_file(
                    model=model,
                    titles_path=titles_path,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    save_path=save_path,
                    min_chars=min_chars,
                )
                print(f"[ok] Saved custom Wikipedia-trained weights to {save_path}")
                continue

            print("[warn] Invalid choice.")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")


def imported_llm_menu() -> None:
    print("\n--- Imported HuggingFace LLM with AST soft-prefix ---")
    try:
        from imported_llm import (
            ImportedAstLLM,
            load_imported_ast_checkpoint,
            predict_imported_truth_text,
            train_imported_ast_llm,
        )

        trained_encoder, tree_dim, source_name, use_ast = choose_tree_embedding_source()
        prefix_tokens = ask_int("Virtual AST prefix tokens", default=4, minimum=1) if use_ast else 1
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return

    default_model = "microsoft/Phi-3-mini-4k-instruct"
    model_name = input(f"Imported model name [{default_model}]> ").strip() or default_model
    print("\nImported tokenizer mode")
    print("1) Pretrained tokenizer")
    print("2) Pretrained tokenizer + added logic special tokens")
    tokenizer_choice = input("select [1]> ").strip() or "1"
    tokenizer_mode = "augmented_logic" if tokenizer_choice == "2" else "pretrained"
    checkpoint_path = ask_optional_path("Imported weights to load (type 'new' for base model)", default=IMPORTED_WEIGHTS_DEFAULT)

    try:
        if checkpoint_path and checkpoint_path.exists():
            model = load_imported_ast_checkpoint(checkpoint_path)
        else:
            if checkpoint_path:
                print(f"[warn] No imported checkpoint found at {checkpoint_path}. Loading base model.")
            model = ImportedAstLLM(
                model_name=model_name,
                tree_embedding_dim=tree_dim,
                tree_prefix_tokens=prefix_tokens,
                tokenizer_mode=tokenizer_mode,
            )

        while True:
            print("\nImported model menu")
            print(f"Embedding source: {source_name}")
            print("1) Tune imported model with AST soft-prefix")
            print("2) Run imported model on labeled statement file")
            print("x) Back")
            choice = input("select> ").strip().lower()

            if choice in {"x", "back", "q", "quit"}:
                break

            if choice == "1":
                input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
                save_path = ask_path("Save imported weights as", default=str(IMPORTED_WEIGHTS_DEFAULT))
                statements = load_labeled_statements(input_path)
                epochs = ask_int("Epochs", default=1, minimum=1)
                batch_size = ask_int("Batch size", default=1, minimum=1)
                lr = ask_float("Learning rate", default=0.0001, minimum=0.0)

                print("\nTraining mode")
                print("1) Adapter only: train AST soft-prefix projector")
                print("2) Adapter + token embeddings")
                print("3) Adapter + unembedding/lm_head")
                print("4) Full imported model")
                mode_choice = input("select [1]> ").strip() or "1"
                train_mode = {
                    "1": "adapter_only",
                    "2": "adapter_and_embeddings",
                    "3": "adapter_and_unembedding",
                    "4": "full",
                }.get(mode_choice, "adapter_only")
                save_base = train_mode != "adapter_only" or tokenizer_mode == "augmented_logic"
                train_encoder = False
                if trained_encoder is not None and use_ast:
                    train_encoder = input("Also train the selected AST encoder during imported-model tuning? (y/N)> ").strip().lower() in {"y", "yes"}

                try:
                    model, _ = train_imported_ast_llm(
                        statements=statements,
                        model_name=model_name,
                        tree_embedding_dim=tree_dim,
                        tree_prefix_tokens=prefix_tokens,
                        model=model,
                        tokenizer_mode=tokenizer_mode,
                        train_mode=train_mode,
                        epochs=epochs,
                        lr=lr,
                        batch_size=batch_size,
                        trained_encoder=trained_encoder,
                        train_tree_encoder=train_encoder,
                        use_ast=use_ast,
                        save_path=save_path,
                        save_base_weights=save_base,
                    )
                except ValueError as exc:
                    print(f"[warn] {exc}")
                    continue
                print(f"[ok] Saved imported AST checkpoint to {save_path}")
                continue

            if choice == "2":
                input_path = ask_path("Evaluation JSONL filename", default=str(GENERATED_DATA_DEFAULT))
                statements = load_labeled_statements(input_path)
                limit = ask_int("How many examples to run?", default=min(10, len(statements)), minimum=1)

                for item in statements[:limit]:
                    generated = predict_imported_truth_text(
                        model=model,
                        formula=item.formula,
                        trained_encoder=trained_encoder,
                        use_ast=use_ast,
                    )
                    print(f"\n{item.name}")
                    print(f"Statement: {item.formula}")
                    print(f"Expected: {'true' if item.label else 'false'}")
                    print(f"Imported LLM says: {generated}")
                continue

            print("[warn] Invalid choice.")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")


def choose_llm_menu() -> None:
    print("\n--- Choose model ---")
    print("1) Imported HuggingFace model")
    print("2) Custom LLM3-style model")
    choice = input("select> ").strip().lower()

    if choice == "1":
        imported_llm_menu()
    elif choice == "2":
        custom_llm_menu()
    else:
        print("[warn] Invalid choice.")


def optional_proof_search_menu() -> None:
    print("\n--- Optional proof-search BFS mode ---")
    name, state = choose_theorem()

    print("\n" + "=" * 60)
    print("SELECTED THEOREM")
    print("=" * 60)
    print(f"Theorem: {name}")
    print(state)
    print_tree_encodings(state)

    print("\nRunning regular BFS baseline...")
    bfs_result = bfs_proof_search(state, max_steps=20)

    print("\nBFS result")
    print("-" * 60)
    print(f"Success: {bfs_result.success}")
    print(f"Visited states: {bfs_result.visited_states}")
    print("Path:", " -> ".join(bfs_result.tactic_path) if bfs_result.success else "none")

    use_llm = input("Also run old Phi tactic-guided BFS? (y/N)> ").strip().lower()
    if use_llm not in {"y", "yes"}:
        return

    from model import PhiTacticModel

    top_k = choose_top_k()
    print("\nLoading Phi-3-mini tactic model...")
    tactic_model = PhiTacticModel()

    print(f"\nRunning LLM-guided top-{top_k} BFS...")
    llm_result = llm_topk_bfs(
        state,
        tactic_model,
        top_k=top_k,
        max_steps=20,
    )

    print("\nLLM-guided result")
    print("-" * 60)
    print(f"Success: {llm_result.success}")
    print(f"Visited states: {llm_result.visited_states}")
    print("Path:", " -> ".join(llm_result.tactic_path) if llm_result.success else "none")
    print_trace(llm_result)


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    while True:
        print("\n=== Logic Truth Evaluation Menu ===")
        print("1) Generate true/false statements to a file")
        print("2) Check labels in a saved statement file")
        print("3) Train the AST embedding truth model")
        print("4) Train the AST embedding/unembedding codec")
        print("5) Use the trained AST embedding truth model")
        print("6) Choose imported or custom LLM")
        print("7) Optional proof-search BFS mode")
        print("q) Quit")

        choice = input("select> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            break
        if choice == "1":
            generate_statement_file_menu()
        elif choice == "2":
            check_statement_file_menu()
        elif choice == "3":
            train_tree_embedding_menu()
        elif choice == "4":
            train_tree_codec_menu()
        elif choice == "5":
            evaluate_tree_embedding_menu()
        elif choice == "6":
            choose_llm_menu()
        elif choice == "7":
            optional_proof_search_menu()
        else:
            print("[warn] Invalid choice.")


if __name__ == "__main__":
    main()
