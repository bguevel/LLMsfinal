from __future__ import annotations

from dataset import make_dataset
from model import PhiTacticModel
from search import bfs_proof_search, llm_topk_bfs


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
        choice = input("Choose theorem number: ").strip()

        try:
            idx = int(choice)
            if 0 <= idx < len(examples):
                return examples[idx]
        except ValueError:
            pass

        print("Invalid choice. Try again.")


def choose_top_k():
    while True:
        choice = input("Choose top-k for LLM search, e.g. 1, 2, 3: ").strip()

        try:
            k = int(choice)
            if k >= 1:
                return k
        except ValueError:
            pass

        print("Invalid top-k. Try again.")


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


def main():
    name, state = choose_theorem()
    top_k = choose_top_k()

    print("\n" + "=" * 60)
    print("SELECTED THEOREM")
    print("=" * 60)
    print(f"Theorem: {name}")
    print(state)

    print("\nRunning regular BFS baseline...")
    bfs_result = bfs_proof_search(state, max_steps=20)

    print("\nBFS result")
    print("-" * 60)
    print(f"Success: {bfs_result.success}")
    print(f"Visited states: {bfs_result.visited_states}")
    if bfs_result.success:
        print("Path:", " -> ".join(bfs_result.tactic_path))
    else:
        print("Path: none")

    print("\nLoading Phi-3-mini...")
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

    if llm_result.success:
        print("Path:", " -> ".join(llm_result.tactic_path))
    else:
        print("Path: none")

    print_trace(llm_result)


if __name__ == "__main__":
    main()