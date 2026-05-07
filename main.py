from __future__ import annotations

import gc
import html
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from custom_llm import (
    EMBEDDING_MODE_LABELS,
    EMBEDDING_MODES,
    load_checkpoint as load_custom_llm_checkpoint,
    make_custom_logic_llm,
    make_default_config,
    normalize_embedding_mode,
    predict_truth_label,
    train_on_labeled_statements,
)
from statement_generation import (
    generate_and_save_labeled_statement_batches,
    load_generation_config,
    load_labeled_statements,
    split_label_counts,
    statement_generation_batches_from_config,
)


WEIGHTS_DIR = Path("weights")
DATA_DIR = Path("data")
GENERATED_DATA_DEFAULT = DATA_DIR / "generated_truth_eval.txt"
GENERATION_CONFIG_DEFAULT = DATA_DIR / "generations.txt"
PLOT_DEFAULT = DATA_DIR / "complexity_accuracy.svg"
LOSS_PLOT_DEFAULT = DATA_DIR / "average_loss_by_mode.svg"


def cleanup_after_training(stage_name: str) -> None:
    print(f"[ok] Finished/left {stage_name}; releasing training resources.")
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    except Exception as exc:
        print(f"[warn] Could not inspect torch cleanup state: {exc}")
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[warn] Torch cleanup after {stage_name} had a warning: {exc}")


def warn_error(stage_name: str, exc: Exception) -> None:
    print(f"[warn] {stage_name} failed: {type(exc).__name__}: {exc}")


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


def ask_path(prompt: str, default: str | Path | None = None) -> Path:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{prompt}{suffix}> ").strip()
        if not raw and default:
            raw = str(default)
        if raw:
            return Path(raw)
        print("[warn] Enter a filename.")


def ask_optional_path(prompt: str, default: str | Path | None = None) -> Path | None:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}> ").strip()
    if not raw:
        return Path(default) if default else None
    if raw.lower() in {"none", "new", "blank"}:
        return None
    return Path(raw)


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        raw = input(f"{prompt}{suffix}> ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("[warn] Enter y or n.")


def ask_training_line_count(total_lines: int) -> int:
    while True:
        raw = input(f"Lines to train on (Enter for full file: {total_lines})> ").strip()
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


def choose_embedding_mode() -> str:
    print("\nCustom model embedding mode")
    for index, mode in enumerate(EMBEDDING_MODES, start=1):
        print(f"{index}) {EMBEDDING_MODE_LABELS[mode]}")

    while True:
        raw = input("select [1]> ").strip() or "1"
        try:
            return normalize_embedding_mode(raw)
        except ValueError as exc:
            print(f"[warn] {exc}")


def default_weights_for_mode(mode: str) -> Path:
    return WEIGHTS_DIR / f"custom_logic_llm_{mode}.pt"


def default_loss_plot_for_mode(mode: str) -> Path:
    return DATA_DIR / f"average_loss_{mode}.svg"


def load_or_create_model_for_mode(mode: str, load_path: Path | None = None):
    if load_path is not None and load_path.exists():
        loaded = load_custom_llm_checkpoint(load_path)
        if loaded is not None:
            model, _ = loaded
            if model.config.embedding_mode != mode:
                print(
                    f"[warn] Loaded checkpoint mode is {model.config.embedding_mode}; "
                    f"using requested mode {mode} with the loaded weights where shapes match."
                )
                model.config.embedding_mode = mode
            print(f"[ok] Loaded {load_path}")
            return model
        print(f"[warn] No checkpoint found at {load_path}. Starting fresh.")

    return make_custom_logic_llm(make_default_config(embedding_mode=mode))


def train_mode_without_prompts(
    mode: str,
    statements,
    epochs: int,
    batch_size: int,
    lr: float,
    load_existing: bool,
) -> list[float]:
    save_path = default_weights_for_mode(mode)
    load_path = save_path if load_existing else None
    model = None
    try:
        print(f"\n--- Training mode: {mode} ---")
        print(f"Embedding setup: {EMBEDDING_MODE_LABELS[mode]}")
        print(f"Save path: {save_path}")
        model = load_or_create_model_for_mode(mode, load_path=load_path)
        losses = train_on_labeled_statements(
            model=model,
            statements=statements,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            save_path=save_path,
        )
        if not save_path.exists():
            raise RuntimeError(f"Expected checkpoint was not written: {save_path}")
        print(f"[ok] Saved {mode} custom LLM checkpoint to {save_path}")
        return losses
    finally:
        model = None
        cleanup_after_training(f"{mode} custom model training")


def running_average_losses(losses: Sequence[float]) -> list[float]:
    averages: list[float] = []
    total = 0.0

    for index, loss in enumerate(losses, start=1):
        total += float(loss)
        averages.append(total / index)

    return averages


def average_loss_series_by_mode(loss_histories: dict[str, Sequence[float]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}

    for mode in EMBEDDING_MODES:
        losses = loss_histories.get(mode)
        if losses:
            series[mode] = running_average_losses(losses)

    for mode, losses in loss_histories.items():
        if mode not in series and losses:
            series[mode] = running_average_losses(losses)

    return series


def write_svg_loss_plot(loss_series: dict[str, Sequence[float]], output_path: Path) -> Path | None:
    if not loss_series:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_updates = max(len(values) for values in loss_series.values())
    all_losses = [float(loss) for values in loss_series.values() for loss in values]
    if not all_losses:
        return None

    y_min = 0.0 if min(all_losses) >= 0.0 else min(all_losses)
    y_max = max(all_losses)
    if y_max <= y_min:
        y_max = y_min + 1.0
    else:
        y_max += (y_max - y_min) * 0.08

    width = 920
    height = 520
    margin_left = 80
    margin_right = 190
    margin_top = 45
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ("#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#4b5563")

    def point_for(update_index: int, value: float) -> tuple[float, float]:
        if max_updates == 1:
            x = margin_left + plot_width / 2
        else:
            x = margin_left + ((update_index - 1) / (max_updates - 1)) * plot_width
        y = margin_top + plot_height * (1 - ((value - y_min) / (y_max - y_min)))
        return x, y

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial" font-size="18" fill="#111827">Average Training Loss by Mode</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
    ]

    for tick_index in range(5):
        tick_value = y_min + (y_max - y_min) * (tick_index / 4)
        y = margin_top + plot_height * (1 - tick_index / 4)
        label = f"{tick_value:.3f}".rstrip("0").rstrip(".") or "0"
        svg.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{label}</text>')

    if max_updates == 1:
        x_tick_values = [1]
    else:
        x_tick_values = sorted({1, max(1, round(max_updates * 0.25)), max(1, round(max_updates * 0.5)), max(1, round(max_updates * 0.75)), max_updates})

    for update_index in x_tick_values:
        x, _ = point_for(update_index, y_min)
        svg.append(f'<line x1="{x:.1f}" y1="{height - margin_bottom}" x2="{x:.1f}" y2="{height - margin_bottom + 5}" stroke="#374151" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{update_index}</text>')

    for series_index, (mode, values) in enumerate(loss_series.items()):
        color = colors[series_index % len(colors)]
        points = [point_for(update_index, float(value)) for update_index, value in enumerate(values, start=1)]
        if len(points) == 1:
            x, y = points[0]
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        else:
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')

        legend_y = margin_top + 20 + series_index * 24
        legend_x = width - margin_right + 28
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        svg.append(
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#111827">'
            f'{html.escape(mode)}</text>'
        )

    svg.append(f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">Running average answer loss</text>')
    svg.append(f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">Training update</text>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def write_loss_plot(loss_histories: dict[str, Sequence[float]], output_path: Path) -> Path | None:
    loss_series = average_loss_series_by_mode(loss_histories)
    if not loss_series:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if output_path.suffix.lower() != ".svg":
            output_path = output_path.with_suffix(".svg")
        return write_svg_loss_plot(loss_series, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.2, 5.2))
    for mode, values in loss_series.items():
        marker = "o" if len(values) <= 20 else None
        plt.plot(range(1, len(values) + 1), values, marker=marker, linewidth=2, label=mode)
    plt.xlabel("Training update")
    plt.ylabel("Running average answer loss")
    plt.title("Average Training Loss by Mode")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


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
    output_path = ask_path("Training/testing data text filename", default=GENERATED_DATA_DEFAULT)
    seed_raw = input("Seed (blank for random)> ").strip()
    seed = int(seed_raw) if seed_raw else None

    statements = generate_and_save_labeled_statement_batches(
        batches=batches,
        output_path=output_path,
        seed=seed,
        progress=True,
    )

    true_count = sum(1 for item in statements if item.label)
    false_count = len(statements) - true_count

    print(f"[ok] Wrote {len(statements)} statements to {output_path}")
    print("     format: formula<TAB>true|false<TAB>complexity")
    print("     complexity levels:")
    for batch in batches:
        level_statements = [item for item in statements if item.complexity == batch.label]
        level_true_count, level_false_count = split_label_counts(batch.total_count, batch.true_fraction)
        print(
            f"     - {batch.label}: total={len(level_statements)}, "
            f"true={level_true_count}, false={level_false_count}, max_depth={batch.max_depth}"
        )
    print(f"     true={true_count}, false={false_count}")


def train_custom_model_menu() -> None:
    print("\n--- Train custom model ---")
    mode = choose_embedding_mode()
    default_save = default_weights_for_mode(mode)
    load_path = ask_optional_path("Custom weights to load (type 'new' for fresh model)", default=default_save)
    input_path = ask_path("Training data text filename", default=GENERATED_DATA_DEFAULT)
    save_path = ask_path("Save custom weights as", default=default_save)

    print("[check] Verifying symbolic truth labels before training...")
    statements = load_labeled_statements(input_path, verify_labels=True)
    if not statements:
        print(f"[warn] No statements found in {input_path}")
        return
    print(f"[ok] Verified {len(statements)} symbolic truth labels.")

    lines_to_train = ask_training_line_count(len(statements))
    selected_statements = statements[:lines_to_train]
    epochs = ask_int("Epochs", default=1, minimum=1)
    batch_size = ask_int("Batch size", default=8, minimum=1)
    lr = ask_float("Learning rate", default=0.0005, minimum=0.0)

    model = None
    try:
        model = load_or_create_model_for_mode(mode, load_path=load_path)
        losses = train_on_labeled_statements(
            model=model,
            statements=selected_statements,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            save_path=save_path,
        )
        if not save_path.exists():
            raise RuntimeError(f"Expected checkpoint was not written: {save_path}")
        print(f"[ok] Saved {mode} custom LLM checkpoint to {save_path}")
        written_loss_plot = write_loss_plot({mode: losses}, default_loss_plot_for_mode(mode))
        if written_loss_plot is not None:
            print(f"[ok] Wrote average loss plot to {written_loss_plot}")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
    except Exception as exc:
        warn_error("Custom model training", exc)
    finally:
        model = None
        cleanup_after_training("custom model training")


def train_all_modes_menu() -> None:
    print("\n--- Train all custom embedding modes ---")
    print("Modes will train sequentially in this order:")
    for index, mode in enumerate(EMBEDDING_MODES, start=1):
        print(f"{index}) {mode}: {EMBEDDING_MODE_LABELS[mode]}")

    input_path = ask_path("Training data text filename", default=GENERATED_DATA_DEFAULT)
    print("[check] Verifying symbolic truth labels before sequential training...")
    statements = load_labeled_statements(input_path, verify_labels=True)
    if not statements:
        print(f"[warn] No statements found in {input_path}")
        return
    print(f"[ok] Verified {len(statements)} symbolic truth labels.")

    lines_to_train = ask_training_line_count(len(statements))
    selected_statements = statements[:lines_to_train]
    epochs = ask_int("Epochs for each mode", default=1, minimum=1)
    batch_size = ask_int("Batch size", default=8, minimum=1)
    lr = ask_float("Learning rate", default=0.0005, minimum=0.0)
    load_existing = ask_yes_no("Load existing per-mode checkpoint first when available?", default=False)

    try:
        loss_histories: dict[str, Sequence[float]] = {}
        for mode in EMBEDDING_MODES:
            loss_histories[mode] = train_mode_without_prompts(
                mode=mode,
                statements=selected_statements,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                load_existing=load_existing,
            )
        written_loss_plot = write_loss_plot(loss_histories, LOSS_PLOT_DEFAULT)
        if written_loss_plot is not None:
            print(f"[ok] Wrote average loss plot to {written_loss_plot}")
        print("\n[ok] Finished sequential training for all custom embedding modes.")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
    except Exception as exc:
        warn_error("Sequential custom model training", exc)


def complexity_sort_key(label: str) -> tuple[int, str]:
    match = re.search(r"\d+", label)
    return (int(match.group(0)) if match else 10**9, label)


def evaluate_custom_model(model, statements) -> tuple[dict[str, dict[str, float]], int, int]:
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for index, statement in enumerate(statements, start=1):
        prediction, confidence = predict_truth_label(model, statement.formula_text)
        correct = prediction == statement.label
        complexity = statement.complexity or "Unknown"
        grouped[complexity]["total"] += 1
        grouped[complexity]["correct"] += int(correct)
        overall_correct += int(correct)
        print(
            f"{index}/{len(statements)} | {complexity} | expected={'true' if statement.label else 'false'} | "
            f"predicted={'true' if prediction else 'false'} | confidence={confidence:.2%}"
        )

    metrics: dict[str, dict[str, float]] = {}
    for complexity, counts in grouped.items():
        total = int(counts["total"])
        correct = int(counts["correct"])
        metrics[complexity] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return metrics, overall_correct, len(statements)


def write_svg_accuracy_plot(metrics: dict[str, dict[str, float]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(metrics, key=complexity_sort_key)
    width = max(640, 110 * max(1, len(labels)))
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 35
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 18
    bar_width = max(24, (plot_width - bar_gap * max(0, len(labels) - 1)) / max(1, len(labels)))

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18" fill="#111827">Complexity vs Accuracy</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
    ]

    for tick in range(0, 101, 25):
        y = margin_top + plot_height * (1 - tick / 100)
        svg.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{tick}%</text>')

    for index, label in enumerate(labels):
        accuracy = float(metrics[label]["accuracy"])
        correct = int(metrics[label]["correct"])
        total = int(metrics[label]["total"])
        bar_height = plot_height * accuracy
        x = margin_left + index * (bar_width + bar_gap)
        y = margin_top + plot_height - bar_height
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>')
        svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{accuracy:.0%}</text>')
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#111827">{html.escape(label)}</text>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 38}" text-anchor="middle" '
            f'font-family="Arial" font-size="11" fill="#6b7280">{correct}/{total}</text>'
        )

    svg.append(f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">Accuracy</text>')
    svg.append(f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">Statement Complexity</text>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def write_accuracy_plot(metrics: dict[str, dict[str, float]], output_path: Path) -> Path:
    labels = sorted(metrics, key=complexity_sort_key)
    accuracies = [metrics[label]["accuracy"] for label in labels]

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if output_path.suffix.lower() != ".svg":
            output_path = output_path.with_suffix(".svg")
        return write_svg_accuracy_plot(metrics, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, len(labels) * 0.8), 4.5))
    plt.bar(labels, accuracies, color="#2563eb")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Statement complexity")
    plt.title("Complexity vs Accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def test_custom_model_menu() -> None:
    print("\n--- Test custom model on a file ---")
    mode = choose_embedding_mode()
    weights_path = ask_path("Custom weights filename", default=default_weights_for_mode(mode))
    input_path = ask_path("Testing data text filename", default=GENERATED_DATA_DEFAULT)
    plot_path = ask_path("Save complexity accuracy plot as", default=PLOT_DEFAULT)

    print("[check] Verifying symbolic truth labels before testing...")
    statements = load_labeled_statements(input_path, verify_labels=True)
    if not statements:
        print(f"[warn] No statements found in {input_path}")
        return
    print(f"[ok] Verified {len(statements)} symbolic truth labels.")

    try:
        loaded = load_custom_llm_checkpoint(weights_path)
        if loaded is None:
            print(f"[warn] No checkpoint found at {weights_path}")
            return
        model, _ = loaded
        if model.config.embedding_mode != mode:
            print(
                f"[warn] Loaded checkpoint mode is {model.config.embedding_mode}; "
                f"evaluating with requested mode {mode}."
            )
            model.config.embedding_mode = mode
        metrics, correct, total = evaluate_custom_model(model, statements)
        overall_accuracy = correct / total if total else 0.0
        print(f"\n[ok] Overall accuracy: {correct}/{total} ({overall_accuracy:.2%})")
        print("By complexity:")
        for label in sorted(metrics, key=complexity_sort_key):
            row = metrics[label]
            print(f"  {label}: {int(row['correct'])}/{int(row['total'])} ({row['accuracy']:.2%})")
        written_plot = write_accuracy_plot(metrics, plot_path)
        print(f"[ok] Wrote complexity accuracy plot to {written_plot}")
    except ModuleNotFoundError as exc:
        print(f"[warn] {exc}")
    except Exception as exc:
        warn_error("Custom model testing", exc)
    finally:
        cleanup_after_training("custom model testing")


def main() -> None:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    while True:
        print("\n=== Logic Truth Evaluation Menu ===")
        print("0) Train all custom embedding modes sequentially")
        print("1) Generate data")
        print("2) Train custom model")
        print("3) Test custom model on a file and plot complexity accuracy")
        print("q) Quit")

        choice = input("select> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            break
        try:
            if choice == "0":
                train_all_modes_menu()
            elif choice == "1":
                generate_statement_file_menu()
            elif choice == "2":
                train_custom_model_menu()
            elif choice == "3":
                test_custom_model_menu()
            else:
                print("[warn] Invalid choice.")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            warn_error("Menu action", exc)
            cleanup_after_training("failed menu action")


if __name__ == "__main__":
    main()
