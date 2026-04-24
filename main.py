from dataset import make_dataset
from model import PhiTacticModel
from search import bfs_proof_search, llm_guided_search


def main():
    examples = make_dataset()
    name, state = examples[2]

    print("=" * 60)
    print("THEOREM:", name)
    print(state)

    print("\nRunning BFS...")
    bfs_result = bfs_proof_search(state, max_steps=20)
    print("BFS success:", bfs_result.success)
    print("BFS path:", " -> ".join(bfs_result.tactic_path))

    print("\nLoading Phi-3-mini...")
    tactic_model = PhiTacticModel()

    print("Running LLM-guided search...")
    llm_result = llm_guided_search(state, tactic_model, max_steps=20)
    print("LLM success:", llm_result.success)
    print("LLM path:", " -> ".join(llm_result.tactic_path))


if __name__ == "__main__":
    main()