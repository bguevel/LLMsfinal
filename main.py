from __future__ import annotations

from pathlib import Path

from custom_llm import (
    make_custom_logic_llm,
    load_checkpoint as load_custom_llm_checkpoint,
    make_default_config,
    predict_truth_text,
    train_custom_llm_on_text_and_wikipedia_files,
    train_custom_llm_on_text_file,
    train_custom_llm_on_wikipedia_file,
    train_on_labeled_statements_with_ast,
)
from statement_generation import (
    DEFAULT_VARIABLES,
    check_statement_truth,
    DEFAULT_STATEMENT_COMPLEXITY,
    generate_and_save_labeled_statement_batches,
    get_statement_complexity,
    load_generation_config,
    load_labeled_statements,
    split_label_counts,
    statement_generation_batches_from_config,
    STATEMENT_COMPLEXITY_LEVELS,
)
from tree_codec import load_tree_codec, train_tree_codec
from tree_encoding import FormulaTreeEncoder
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
GENERATION_CONFIG_DEFAULT = DATA_DIR / "generations.txt"
WIKIPEDIA_TITLES_DEFAULT = DATA_DIR / "wikipedia_titles_500.txt"


def ask_int(
    prompt: str,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
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
            if maximum is not None and value > maximum:
                print(f"[warn] Enter an integer <= {maximum}.")
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


def ask_path_list(prompt: str) -> list[Path]:
    raw = input(f"{prompt}> ").strip()
    if not raw or raw.lower() in {"none", "skip"}:
        return []
    paths = [item.strip().strip('"') for item in raw.split(",")]
    return [Path(item) for item in paths if item]


def ask_training_line_count(total_lines: int) -> int:
    while True:
        raw = input(f"JSONL lines to train on (Enter for full file: {total_lines})> ").strip()
        if not raw:
            return total_lines
        try:
            value = int(raw)
        except ValueError:
            print("[warn] Enter an integer, or press Enter for the full file.")
            continue
        if value < 1:
            print("[warn] Enter an integer >= 1.")
            continue
        if value > total_lines:
            print(f"[warn] Enter an integer <= {total_lines}, or press Enter for the full file.")
            continue
        return value


def ask_variable_names(prompt: str, default: tuple[str, ...] = DEFAULT_VARIABLES) -> tuple[str, ...]:
    while True:
        raw = input(f"{prompt} [{' '.join(default)}]> ").strip()
        variables = tuple(raw.split()) if raw else default
        if variables:
            return variables
        print("[warn] Enter at least one variable name.")


def choose_statement_complexity() -> tuple[str, int, tuple[str, ...]]:
    default_complexity = get_statement_complexity(DEFAULT_STATEMENT_COMPLEXITY)
    custom_index = len(STATEMENT_COMPLEXITY_LEVELS) + 1

    while True:
        print("\nStatement complexity")
        for i, option in enumerate(STATEMENT_COMPLEXITY_LEVELS, start=1):
            print(
                f"{i}) {option.label}: {option.description} "
                f"(depth={option.max_depth}, vars={', '.join(option.variables)})"
            )
        print(f"{custom_index}) Custom")

        choice = input(f"select [{default_complexity.key}]> ").strip().lower()
        if not choice:
            option = default_complexity
            return option.label, option.max_depth, option.variables

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(STATEMENT_COMPLEXITY_LEVELS):
                option = STATEMENT_COMPLEXITY_LEVELS[index - 1]
                return option.label, option.max_depth, option.variables
            if index == custom_index:
                max_depth = ask_int("Max random subformula depth", default=default_complexity.max_depth, minimum=0)
                variables = ask_variable_names("Variable names")
                return "Custom", max_depth, variables

        if choice in {"custom", "c"}:
            max_depth = ask_int("Max random subformula depth", default=default_complexity.max_depth, minimum=0)
            variables = ask_variable_names("Variable names")
            return "Custom", max_depth, variables

        try:
            option = get_statement_complexity(choice)
            return option.label, option.max_depth, option.variables
        except ValueError:
            print("[warn] Choose a complexity number, name, or custom.")


def generate_statement_file_menu() -> None:
    print("\n--- Generate true/false statements ---")
    try:
        config = load_generation_config(GENERATION_CONFIG_DEFAULT)
    except OSError as exc:
        print(f"[warn] Could not read generation config at {GENERATION_CONFIG_DEFAULT}: {exc}")
        return
    except ValueError as exc:
        print(f"[warn] Could not parse generation config at {GENERATION_CONFIG_DEFAULT}: {exc}")
        return

    batches = statement_generation_batches_from_config(config)
    output_path = ask_path("Training/testing data JSONL filename", default=str(GENERATED_DATA_DEFAULT))
    seed_raw = input("Seed (blank for random)> ").strip()
    seed = int(seed_raw) if seed_raw else None

    statements = generate_and_save_labeled_statement_batches(
        batches=batches,
        output_path=output_path,
        seed=seed,
    )
    level_summaries = []
    for batch in batches:
        true_count, false_count = split_label_counts(batch.total_count, batch.true_fraction)
        level_summaries.append(
            (
                batch.label,
                batch.total_count,
                true_count,
                false_count,
                batch.max_depth,
                batch.variables,
            )
        )

    true_count = sum(1 for item in statements if item.label)
    false_count = len(statements) - true_count
    correct, verified_total = verify_symbolic_labels(statements)
    print(f"[ok] Wrote {len(statements)} statements to {output_path}")
    print("     complexity levels:")
    for label, level_total, level_true_count, level_false_count, max_depth, variables in level_summaries:
        print(
            f"     - {label.lower()}: total={level_total}, true={level_true_count}, "
            f"false={level_false_count}, max_depth={max_depth}, variables={', '.join(variables)}"
        )
    print(f"     true={true_count}, false={false_count}")
    print(f"     symbolic labels verified={correct}/{verified_total}")


def check_statement_file_menu() -> None:
    print("\n--- Check saved statement file ---")
    input_path = ask_path("Input JSONL filename", default=str(GENERATED_DATA_DEFAULT))
    statements = load_labeled_statements(input_path, verify_labels=False)
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
    statements = load_labeled_statements(input_path, verify_labels=False)

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
    statements = load_labeled_statements(input_path, verify_labels=False)

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
    print("\n--- Run trained AST embedding truth model on a file ---")
    model_path = ask_path("Trained embedding model filename", default=str(TRUTH_ENCODER_DEFAULT))
    input_path = ask_path("Statement JSONL/JSON filename to run", default=str(GENERATED_DATA_DEFAULT))

    try:
        model = load_truth_embedding_model(model_path)
        statements = load_labeled_statements(input_path, verify_labels=False)
        metrics = evaluate_truth_embedding_model(model, statements)
        print(f"[ok] Evaluated {metrics['total']} statements")
        print(f"Accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.2%})")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")


def train_embedding_models_menu() -> None:
    while True:
        print("\n--- Train Embedding Models ---")
        print("1) Train truth-classifier AST embedding")
        print("2) Train embedding/unembedding codec")
        print("x) Back")
        choice = input("select> ").strip().lower()

        if choice in {"x", "back", "q", "quit"}:
            break
        if choice == "1":
            train_tree_embedding_menu()
        elif choice == "2":
            train_tree_codec_menu()
        else:
            print("[warn] Invalid choice.")


def run_trained_model_file_menu() -> None:
    while True:
        print("\n--- Run Trained Model On File ---")
        print("1) Trained AST embedding truth model")
        print("2) Imported or custom LLM checkpoint")
        print("x) Back")
        choice = input("select> ").strip().lower()

        if choice in {"x", "back", "q", "quit"}:
            break
        if choice == "1":
            evaluate_tree_embedding_menu()
        elif choice == "2":
            choose_llm_menu()
        else:
            print("[warn] Invalid choice.")


def choose_statement_model_kind() -> str | None:
    print("\n--- Choose model ---")
    print("1) Imported HuggingFace model")
    print("2) Custom LLM3-style model")
    choice = input("select> ").strip().lower()

    if choice == "1":
        return "imported"
    if choice == "2":
        return "custom"

    print("[warn] Invalid choice.")
    return None


def choose_tree_embedding_source():
    print("\nEmbedding source for custom/imported statement models")
    print("0) REGULAR TOKEN EMBEDDINGS ONLY: token + position embeddings, no parallel AST prefix")
    print("1) PARALLEL AST: fixed hash tree embedding")
    print("2) PARALLEL AST: untrained recursive tree encoder")
    print("3) PARALLEL AST: trained truth-classifier encoder")
    print("4) PARALLEL AST: trained embedding/unembedding codec encoder")
    print(
        "Difference: regular embeddings come from the tokenizer sequence only; "
        "parallel AST embeddings add a separate formula-tree vector as soft-prefix tokens."
    )
    choice = input("select [1]> ").strip().lower() or "1"

    if choice == "0":
        return None, 256, "regular-token-only", False

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


def train_custom_statement_model_menu() -> list | None:
    print("\n--- Train Custom Statement Model ---")
    try:
        trained_encoder, tree_dim, source_name, use_ast = choose_tree_embedding_source()
        prefix_tokens = ask_int("Virtual AST prefix tokens", default=4, minimum=1) if use_ast else 1
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return None

    tokenizer_kind = "logic"
    print("\nCustom tokenizer: logic-aware")

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
                tokenizer_kind = getattr(model.tokenizer, "tokenizer_kind", "word")
                if tokenizer_kind != "logic":
                    print(
                        f"[warn] {load_path} uses tokenizer={tokenizer_kind}; "
                        "starting fresh with the logic-aware tokenizer."
                    )
                    tokenizer_kind = "logic"
                    model = make_custom_logic_llm(
                        make_default_config(tree_embedding_dim=tree_dim, tree_prefix_tokens=prefix_tokens),
                        tokenizer_kind=tokenizer_kind,
                    )
                else:
                    print(f"[ok] Loaded {load_path}")
        else:
            model = make_custom_logic_llm(
                make_default_config(tree_embedding_dim=tree_dim, tree_prefix_tokens=prefix_tokens),
                tokenizer_kind=tokenizer_kind,
            )

        print(f"\nSelected embedding: {source_name}")
        input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
        default_save = CUSTOM_WEIGHTS_DEFAULT if use_ast else WEIGHTS_DIR / "custom_logic_llm_regular_only.pt"
        save_path = ask_path("Save custom weights as", default=str(default_save))
        statements = load_labeled_statements(input_path, verify_labels=False)
        lines_to_train = ask_training_line_count(len(statements))
        selected_statements = statements[:lines_to_train]
        epochs = ask_int("Epochs per line", default=1, minimum=1)
        lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
        train_encoder = False
        if trained_encoder is not None and use_ast:
            train_encoder = input("Also train the selected AST encoder during LLM tuning? (y/N)> ").strip().lower() in {"y", "yes"}

        train_on_labeled_statements_with_ast(
            model=model,
            statements=selected_statements,
            epochs=epochs,
            lr=lr,
            batch_size=1,
            trained_encoder=trained_encoder,
            train_tree_encoder=train_encoder,
            use_ast=use_ast,
            save_path=save_path,
        )
        label = "AST-conditioned" if use_ast else "regular-token-only"
        print(f"[ok] Saved {label} custom LLM checkpoint to {save_path}")
        return selected_statements
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return None


def choose_imported_train_mode(use_ast: bool) -> str:
    print("\nTraining mode")
    if use_ast:
        print("1) Adapter only: train AST soft-prefix projector")
        print("2) Adapter + token embeddings")
        print("3) Adapter + unembedding/lm_head")
        print("4) Full imported model")
        mode_choice = input("select [1]> ").strip() or "1"
        return {
            "1": "adapter_only",
            "2": "adapter_and_embeddings",
            "3": "adapter_and_unembedding",
            "4": "full",
        }.get(mode_choice, "adapter_only")

    print("1) Token embeddings only")
    print("2) Unembedding/lm_head only")
    print("3) Full imported model")
    mode_choice = input("select [3]> ").strip() or "3"
    return {
        "1": "adapter_and_embeddings",
        "2": "adapter_and_unembedding",
        "3": "full",
    }.get(mode_choice, "full")


def train_imported_statement_model_menu() -> list | None:
    print("\n--- Train Imported Statement Model ---")
    try:
        from imported_llm import (
            ImportedAstLLM,
            load_imported_ast_checkpoint,
            train_imported_ast_llm,
        )

        trained_encoder, tree_dim, source_name, use_ast = choose_tree_embedding_source()
        prefix_tokens = ask_int("Virtual AST prefix tokens", default=4, minimum=1) if use_ast else 1
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return None

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

        print(f"\nSelected embedding: {source_name}")
        input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
        save_path = ask_path("Save imported weights as", default=str(IMPORTED_WEIGHTS_DEFAULT))
        statements = load_labeled_statements(input_path, verify_labels=False)
        lines_to_train = ask_training_line_count(len(statements))
        selected_statements = statements[:lines_to_train]
        epochs = ask_int("Epochs per line", default=1, minimum=1)
        lr = ask_float("Learning rate", default=0.0001, minimum=0.0)
        train_mode = choose_imported_train_mode(use_ast=use_ast)
        save_base = train_mode != "adapter_only" or tokenizer_mode == "augmented_logic"
        train_encoder = False
        if trained_encoder is not None and use_ast:
            train_encoder = input("Also train the selected AST encoder during imported-model tuning? (y/N)> ").strip().lower() in {"y", "yes"}

        try:
            train_imported_ast_llm(
                statements=selected_statements,
                model_name=model_name,
                tree_embedding_dim=tree_dim,
                tree_prefix_tokens=prefix_tokens,
                model=model,
                tokenizer_mode=tokenizer_mode,
                train_mode=train_mode,
                epochs=epochs,
                lr=lr,
                batch_size=1,
                trained_encoder=trained_encoder,
                train_tree_encoder=train_encoder,
                use_ast=use_ast,
                save_path=save_path,
                save_base_weights=save_base,
            )
        except ValueError as exc:
            print(f"[warn] {exc}")
            return None

        label = "AST-conditioned" if use_ast else "regular-token-only"
        print(f"[ok] Saved {label} imported LLM checkpoint to {save_path}")
        return selected_statements
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return None


def train_both_embedding_models_on_statements(statements: list) -> None:
    if not statements:
        print("[warn] No statements were selected for embedding training.")
        return

    print("\n--- Train Both Standalone AST Embedding Models ---")
    print(f"Using the same selected training data: {len(statements)} JSONL lines")
    truth_save_path = ask_path("Save truth-classifier AST embedding as", default=str(TRUTH_ENCODER_DEFAULT))
    codec_save_path = ask_path("Save AST embedding/unembedding codec as", default=str(TREE_CODEC_DEFAULT))
    dim = ask_int("Embedding dimension for both models", default=128, minimum=8)
    epochs = ask_int("Embedding epochs", default=20, minimum=1)
    truth_batch_size = ask_int("Truth-classifier batch size", default=16, minimum=1)
    codec_batch_size = ask_int("Codec batch size", default=8, minimum=1)
    lr = ask_float("Embedding learning rate", default=0.001, minimum=0.0)

    try:
        truth_model, _ = train_truth_embedding_model(
            statements=statements,
            dim=dim,
            epochs=epochs,
            batch_size=truth_batch_size,
            lr=lr,
            save_path=truth_save_path,
        )
        metrics = evaluate_truth_embedding_model(truth_model, statements)
        print(f"[ok] Saved truth-classifier AST embedding to {truth_save_path}")
        print(f"Training-file accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.2%})")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
    except Exception as exc:
        print(f"[warn] Truth-classifier AST embedding training failed, continuing: {exc}")

    try:
        train_tree_codec(
            statements=statements,
            dim=dim,
            epochs=epochs,
            batch_size=codec_batch_size,
            lr=lr,
            save_path=codec_save_path,
        )
        print(f"[ok] Saved AST embedding/unembedding codec to {codec_save_path}")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
    except Exception as exc:
        print(f"[warn] AST embedding/unembedding codec training failed: {exc}")


def train_statement_model_then_embeddings_menu() -> None:
    print("\n--- Train Statement Model, Then Embeddings ---")
    model_kind = choose_statement_model_kind()

    selected_statements = None
    if model_kind == "imported":
        selected_statements = train_imported_statement_model_menu()
    elif model_kind == "custom":
        selected_statements = train_custom_statement_model_menu()

    if selected_statements:
        train_both_embedding_models_on_statements(selected_statements)


def train_statement_model_menu() -> None:
    print("\n--- Train Statement Model ---")
    print("0) Train chosen statement model, then both standalone AST embedding models")
    print("1) Imported HuggingFace model")
    print("2) Custom LLM3-style model")
    choice = input("select> ").strip().lower()

    if choice == "0":
        train_statement_model_then_embeddings_menu()
    elif choice == "1":
        train_imported_statement_model_menu()
    elif choice == "2":
        train_custom_statement_model_menu()
    else:
        print("[warn] Invalid choice.")



def custom_llm_menu() -> None:
    print("\n--- Custom LLM3-style model ---")
    try:
        trained_encoder, tree_dim, source_name, use_ast = choose_tree_embedding_source()
        prefix_tokens = ask_int("Virtual AST prefix tokens", default=4, minimum=1) if use_ast else 1
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
        return

    tokenizer_kind = "logic"
    print("\nCustom tokenizer: logic-aware")

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
                tokenizer_kind = getattr(model.tokenizer, "tokenizer_kind", "word")
                if tokenizer_kind != "logic":
                    print(
                        f"[warn] {load_path} uses tokenizer={tokenizer_kind}; "
                        "starting fresh with the logic-aware tokenizer."
                    )
                    tokenizer_kind = "logic"
                    model = make_custom_logic_llm(
                        make_default_config(tree_embedding_dim=tree_dim, tree_prefix_tokens=prefix_tokens),
                        tokenizer_kind=tokenizer_kind,
                    )
                else:
                    print(f"[ok] Loaded {load_path}")
        else:
            model = make_custom_logic_llm(
                make_default_config(tree_embedding_dim=tree_dim, tree_prefix_tokens=prefix_tokens),
                tokenizer_kind=tokenizer_kind,
            )

        while True:
            selected_embedding_label = (
                "PARALLEL AST embedding"
                if use_ast
                else "REGULAR TOKEN EMBEDDINGS ONLY"
            )
            print("\nCustom LLM menu")
            print(f"Embedding source: {source_name}")
            print(f"Tokenizer: {tokenizer_kind}")
            print("\nEmbedding choices")
            print("1) PARALLEL AST EMBEDDING: token embeddings plus selected AST soft-prefix")
            print("3) REGULAR TOKEN EMBEDDINGS ONLY: token + position embeddings, no AST prefix")
            print(
                "   Difference: option 1 gives attention a separate structural formula vector; "
                "option 3 learns only from the tokenized formula text."
            )
            print("\nTraining options")
            print(f"1) Train labeled statements with SELECTED setting: {selected_embedding_label}")
            print("2) Run on labeled statement file")
            print("3) Train labeled statements with REGULAR TOKEN EMBEDDINGS ONLY baseline")
            print("4) Train custom model on a plain text file")
            print("5) Train custom model on Wikipedia articles from a title/URL file")
            print("6) Train custom model on plain text plus Wikipedia articles")
            print("x) Back")
            choice = input("select> ").strip().lower()

            if choice in {"x", "back", "q", "quit"}:
                break

            if choice == "1":
                input_path = ask_path("Training JSONL filename", default=str(GENERATED_DATA_DEFAULT))
                save_path = ask_path("Save custom weights as", default=str(CUSTOM_WEIGHTS_DEFAULT))
                statements = load_labeled_statements(input_path, verify_labels=False)
                lines_to_train = ask_training_line_count(len(statements))
                selected_statements = statements[:lines_to_train]
                epochs = ask_int("Epochs per line", default=1, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                train_encoder = False
                if trained_encoder is not None and use_ast:
                    train_encoder = input("Also train the selected AST encoder during LLM tuning? (y/N)> ").strip().lower() in {"y", "yes"}
                train_on_labeled_statements_with_ast(
                    model=model,
                    statements=selected_statements,
                    epochs=epochs,
                    lr=lr,
                    batch_size=1,
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
                statements = load_labeled_statements(input_path, verify_labels=False)
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
                save_path = ask_path("Save custom weights as", default=str(WEIGHTS_DIR / "custom_logic_llm_regular_only.pt"))
                statements = load_labeled_statements(input_path, verify_labels=False)
                lines_to_train = ask_training_line_count(len(statements))
                selected_statements = statements[:lines_to_train]
                epochs = ask_int("Epochs per line", default=1, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                train_on_labeled_statements_with_ast(
                    model=model,
                    statements=selected_statements,
                    epochs=epochs,
                    lr=lr,
                    batch_size=1,
                    trained_encoder=None,
                    use_ast=False,
                    save_path=save_path,
                )
                print(f"[ok] Saved regular-token-embedding custom LLM checkpoint to {save_path}")
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
                titles_path = ask_path("Wikipedia titles/URLs file", default=str(WIKIPEDIA_TITLES_DEFAULT))
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

            if choice == "6":
                text_paths = ask_path_list("Plain text training files, comma-separated (blank to skip)")
                titles_path = ask_optional_path(
                    "Wikipedia titles/URLs file (type 'none' to skip)",
                    default=WIKIPEDIA_TITLES_DEFAULT,
                )
                if not text_paths and titles_path is None:
                    print("[warn] Choose at least one plain text file or a Wikipedia titles file.")
                    continue

                save_path = ask_path("Save mixed text-trained weights as", default=str(WEIGHTS_DIR / "custom_logic_llm_mixed_text.pt"))
                epochs = ask_int("Epochs", default=3, minimum=1)
                batch_size = ask_int("Batch size", default=8, minimum=1)
                seq_len = ask_int("Sequence length", default=128, minimum=1)
                lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
                min_chars = ask_int("Minimum Wikipedia article characters", default=200, minimum=1)
                train_custom_llm_on_text_and_wikipedia_files(
                    model=model,
                    text_paths=text_paths,
                    titles_path=titles_path,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    save_path=save_path,
                    min_chars=min_chars,
                )
                print(f"[ok] Saved mixed text-trained weights to {save_path}")
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
                statements = load_labeled_statements(input_path, verify_labels=False)
                lines_to_train = ask_training_line_count(len(statements))
                selected_statements = statements[:lines_to_train]
                epochs = ask_int("Epochs per line", default=1, minimum=1)
                lr = ask_float("Learning rate", default=0.0001, minimum=0.0)

                train_mode = choose_imported_train_mode(use_ast=use_ast)
                save_base = train_mode != "adapter_only" or tokenizer_mode == "augmented_logic"
                train_encoder = False
                if trained_encoder is not None and use_ast:
                    train_encoder = input("Also train the selected AST encoder during imported-model tuning? (y/N)> ").strip().lower() in {"y", "yes"}

                try:
                    model, _ = train_imported_ast_llm(
                        statements=selected_statements,
                        model_name=model_name,
                        tree_embedding_dim=tree_dim,
                        tree_prefix_tokens=prefix_tokens,
                        model=model,
                        tokenizer_mode=tokenizer_mode,
                        train_mode=train_mode,
                        epochs=epochs,
                        lr=lr,
                        batch_size=1,
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
                statements = load_labeled_statements(input_path, verify_labels=False)
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
    model_kind = choose_statement_model_kind()

    if model_kind == "imported":
        imported_llm_menu()
    elif model_kind == "custom":
        custom_llm_menu()


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    while True:
        print("\n=== Logic Truth Evaluation Menu ===")
        print("1) Generate true/false statements to a file")
        print("2) Check labels in a saved statement file")
        print("3) Train embedding models")
        print("4) Train statement model")
        print("5) Run a trained model on a statement file")
        print("6) Advanced imported/custom LLM menu")
        print("q) Quit")

        choice = input("select> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            break
        if choice == "1":
            generate_statement_file_menu()
        elif choice == "2":
            check_statement_file_menu()
        elif choice == "3":
            train_embedding_models_menu()
        elif choice == "4":
            train_statement_model_menu()
        elif choice == "5":
            run_trained_model_file_menu()
        elif choice == "6":
            choose_llm_menu()
        else:
            print("[warn] Invalid choice.")


if __name__ == "__main__":
    main()
